import pytest
import torch

from lmdeploy.pytorch.backends.lora import AdapterInfo
from lmdeploy.pytorch.kernels.cuda.dora import factored_dora_norm, pack_dora_weight_norm
from lmdeploy.pytorch.nn.linear.lora import LoRA


class TestFactoredDoraNorm:
    @pytest.fixture
    def ranks(self):
        yield torch.tensor([2, 4])

    @pytest.fixture
    def rank_offsets(self, ranks):
        yield ranks.cumsum(0) - ranks

    @pytest.fixture
    def scaling(self, ranks):
        yield torch.arange(ranks.size(0)) + 1.0

    def test_factored_norm_matches_naive(self, ranks, rank_offsets, scaling):
        """The factored norm must equal the norm of the materialized W + s*BA."""
        torch.manual_seed(0)
        out_features, in_features = 16, 32
        weight = torch.randn(out_features, in_features, dtype=torch.float32)
        lora_a = torch.randn(int(ranks.sum()), in_features, dtype=torch.float32)
        lora_b = torch.randn(int(ranks.sum()), out_features, dtype=torch.float32)

        for adapter_id in range(ranks.numel()):
            rank = ranks[adapter_id].item()
            s = scaling[adapter_id].item()
            r_start = rank_offsets[adapter_id].item()
            ba = lora_b[r_start : r_start + rank].t() @ lora_a[r_start : r_start + rank]
            naive = (weight + s * ba).norm(dim=1)

            factored = factored_dora_norm(weight, lora_a, lora_b, s, rank, adapter_id, rank_offsets)
            torch.testing.assert_close(naive, factored, rtol=1e-4, atol=1e-4)

    def test_pack_matches_dora_reference(self, ranks, rank_offsets, scaling):
        """pack_dora_weight_norm must reproduce the reference DoRA forward."""
        torch.manual_seed(0)
        out_features, in_features = 8, 12
        weight = torch.randn(out_features, in_features, dtype=torch.float32)
        lora_a = torch.randn(int(ranks.sum()), in_features, dtype=torch.float32)
        lora_b = torch.randn(int(ranks.sum()), out_features, dtype=torch.float32)
        lora_magnitude = torch.rand(int(ranks.sum()), out_features, dtype=torch.float32) + 0.5

        weight_scaling, _ = pack_dora_weight_norm(weight, lora_a, lora_b, ranks, scaling, rank_offsets, lora_magnitude)

        x = torch.randn(5, in_features, dtype=torch.float32)
        base_out = x @ weight.t()

        for adapter_id in range(ranks.numel()):
            rank = ranks[adapter_id].item()
            r_start = rank_offsets[adapter_id].item()
            s = scaling[adapter_id].item()

            # reference: m * (W + s*BA) / ||W + s*BA||_c
            adapted = weight + s * lora_b[r_start : r_start + rank].t() @ lora_a[r_start : r_start + rank]
            norm = adapted.norm(dim=1, keepdim=True).clamp(min=1e-6)
            m = lora_magnitude[r_start : r_start + rank].sum(dim=0, keepdim=True).t()
            ref = x @ (m * adapted / norm).t()

            got = (
                base_out * weight_scaling[adapter_id][None, :]
                + (x @ lora_a[r_start : r_start + rank].t() @ lora_b[r_start : r_start + rank].t())
                * s
                * weight_scaling[adapter_id][None, :]
            )

            torch.testing.assert_close(ref, got, rtol=1e-4, atol=1e-4)


class TestLoRADoraWiring:
    """LoRA is the layer the patched model builds; check its dora wiring."""

    @pytest.fixture
    def lora_args(self):
        ranks = torch.tensor([4])
        scalings = torch.tensor([2.0])
        in_features, out_features = 12, 8
        lora_a = torch.randn(int(ranks.sum()), in_features, dtype=torch.float16)
        lora_b = torch.randn(int(ranks.sum()), out_features, dtype=torch.float16)
        yield ranks, scalings, in_features, out_features, lora_a, lora_b

    def test_finalize_dora_sets_weight_scaling(self, lora_args, monkeypatch):
        ranks, scalings, in_features, out_features, lora_a, lora_b = lora_args

        class _StubImpl:
            def forward(self, *args, **kwargs):
                return None

        class _StubBuilder:
            @staticmethod
            def build():
                return _StubImpl()

        import lmdeploy.pytorch.nn.linear.lora as lora_mod

        monkeypatch.setattr(
            lora_mod,
            "get_backend",
            lambda: type("B", (), {"get_layer_impl_builder": staticmethod(lambda op: _StubBuilder())}),
        )
        monkeypatch.setattr(lora_mod, "OpType", type("OpType", (), {"LoRA": "lora"}))

        weight = torch.randn(out_features, in_features, dtype=torch.float32)
        magnitude = torch.rand(int(ranks.sum()), out_features, dtype=torch.float32) + 0.5

        lora = LoRA(
            in_features,
            out_features,
            ranks=ranks,
            scalings=scalings,
            lora_a=lora_a,
            lora_b=lora_b,
            base_slice=slice(0, out_features),
            use_dora=True,
            weight=weight,
        )
        lora.register_parameter("lora_magnitude", torch.nn.Parameter(magnitude, requires_grad=False))

        # plain lora layers must not expose a weight scaling
        plain = AdapterInfo(
            in_features=in_features,
            out_features=out_features,
            ranks=ranks,
            scalings=scalings,
            base_slice=slice(0, out_features),
        )
        assert plain.weight_scaling is None

        lora.finalize_dora()
        assert lora.adapter_info.weight_scaling is not None
        assert lora.adapter_info.weight_scaling.shape == (ranks.numel(), out_features)

        # reference value for the single adapter
        m_norm = magnitude.sum(dim=0)
        d = (weight + scalings[0].item() * (lora_b.to(torch.float32).t() @ lora_a.to(torch.float32))).norm(dim=1)
        torch.testing.assert_close(m_norm / d, lora.adapter_info.weight_scaling[0], rtol=1e-3, atol=1e-3)

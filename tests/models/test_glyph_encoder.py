import torch

from poor_word.models.glyph_encoder import GlyphEncoder


def test_encoder_returns_unit_normalized_embeddings() -> None:
    model = GlyphEncoder(embedding_dim=256, pretrained=False).eval()

    with torch.inference_mode():
        output = model(torch.rand(2, 3, 96, 96))

    assert output.shape == (2, 256)
    assert torch.allclose(output.norm(dim=1), torch.ones(2), atol=1e-5)

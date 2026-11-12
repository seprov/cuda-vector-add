#pragma once

#include "fused.cuh"

// Share the fused composite's single-result buffer layout and cleanup.
class GaloisComposite final : public FusedComposite
{
public:
  using FusedComposite::FusedComposite;

  void husk() override;
};

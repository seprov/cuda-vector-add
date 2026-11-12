#pragma once

#include "composite.cuh"

class SplitComposite final : public Composite
{
public:
  using Composite::Composite;

  // Also clean up when the caller leaves scope without calling stubble_burn().
  ~SplitComposite() override { stubble_burn(); }

  void husk() override;
  void stubble_burn() override;

private:
  int *d_vec1_ = nullptr;
  int *d_vec2_ = nullptr;
};

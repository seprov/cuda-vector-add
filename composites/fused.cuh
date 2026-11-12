#pragma once

#include "composite.cuh"

class FusedComposite : public Composite
{
public:
  using Composite::Composite;

  // Also clean up when the caller leaves scope without calling stubble_burn().
  ~FusedComposite() override { stubble_burn(); }

  void husk() override;
  void stubble_burn() override;
};

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <type_traits>
#include "../composites/fused.cuh"
#include "../composites/split.cuh"
#include "../husks/fused.cu"
#include "../husks/split.cu"
#include "../stubble_burning/fused.cu"
#include "../stubble_burning/split.cu"

static_assert(!std::is_copy_constructible_v<FusedComposite>);
static_assert(!std::is_copy_assignable_v<SplitComposite>);

template <typename T>
bool check_lifecycle(unsigned int seed)
{
  // Force multiple grid-stride iterations and cross the CPU chunk boundary.
  // Use the base pointer to exercise virtual calls and destructor dispatch.
  std::unique_ptr<Composite> obj = std::make_unique<T>(65537, seed, 1, 128);
  obj->stubble_burn();
  if (obj->verify_result())
    return false;

  obj->husk();
  if (!obj->verify_result())
    return false;

  // Rerun while the previous result is still owned by this composite.
  obj->husk();
  if (!obj->verify_result())
    return false;

  obj->stubble_burn();
  obj->stubble_burn();
  if (obj->verify_result())
    return false;

  // Reuse after explicit cleanup, then let the destructor perform cleanup.
  obj->husk();
  return obj->verify_result();
}

int main()
{
  const unsigned int seeds[] = {0, 111111,
                                std::numeric_limits<unsigned int>::max()};
  for (unsigned int seed : seeds)
  {
    if (!check_lifecycle<FusedComposite>(seed) ||
        !check_lifecycle<SplitComposite>(seed))
    {
      std::fprintf(stderr, "Composite lifecycle test failed for seed %u.\n", seed);
      return EXIT_FAILURE;
    }
  }
  std::puts("Composite lifecycle tests passed (12 verification-order errors expected).");
  return EXIT_SUCCESS;
}

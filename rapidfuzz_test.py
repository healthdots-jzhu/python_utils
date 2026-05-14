from rapidfuzz import fuzz, process
import numpy as np

def test_fuzz():
    assert fuzz.ratio("I am Jason", "I am jazon") == 80
    daysRatio = fuzz.ratio("World cup 2026 starts in ~30 days", "world cup 2022 started 33 days ago")
    print(f"Days similarity: {daysRatio}")
    assert daysRatio > 70
    print(f"Doctor similarity: {fuzz.ratio('dr Chow', 'dr. Chow')}")

def test_process():
    choices = ["hello world", "hallo world", "hello", "world", "Jason"]
    assert process.extractOne("hello world", choices)[0] == "hello world"
    assert process.extractOne("hallo world", choices)[0] == "hallo world"
    assert process.extractOne("hello", choices)[0] == "hello"
    assert process.extractOne("world", choices)[0] == "world"
    assert process.extractOne("ja30n", choices)[0] == "Jason"

if __name__ == "__main__":
    test_fuzz()
    test_process()
    rng = np.random.default_rng(seed=42)
    print(f"Random ratios: {rng.random(5)}")
    print(f"Random integers: {rng.integers(low=12, high=46, size=5)}")
    print("All tests passed!")

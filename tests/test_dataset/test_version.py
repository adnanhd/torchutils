from torchutils.utils import Version

def test_version_lt():
    assert Version(1, 1, 2) < Version(1, 1 ,3)
    assert Version(1, 2, 1) < Version(1, 3 ,1)
    assert Version(2, 1, 1) < Version(3, 1 ,1)

def test_version_gt():
    assert Version(1, 1 ,3) > Version(1, 1, 2) 
    assert Version(1, 3 ,1) > Version(1, 2, 1) 
    assert Version(3, 1 ,1) > Version(2, 1, 1) 

def test_version_eq():
    assert Version(1, 2, 3) == Version(1, 2, 3)
    assert Version(1, 2, 2) != Version(1, 2, 3)
    assert Version(1, 1, 3) != Version(1, 2, 3)
    assert Version(4, 2, 3) != Version(1, 2, 3)

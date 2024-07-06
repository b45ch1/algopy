if __name__ == '__main__':

    try:
        import pytest
    except:
        print('Please install pytest.')

    retcode = pytest.main(["-x", "algopy"])




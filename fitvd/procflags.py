NO_ATTEMPT=2**0
NO_DATA=2**1
IMAGE_FLAGS=2**2
PSF_FAILURE=2**3
OBJ_FAILURE=2**4

FLAG_MAP={
    'ok':0,
    0:'ok',
    'no_attempt':NO_ATTEMPT,
    NO_ATTEMPT:'no_attempt',

    'no_data':NO_DATA,
    NO_DATA:'no_data',

    'image_flags':IMAGE_FLAGS,
    IMAGE_FLAGS:'image_flags',

    'psf_failure': PSF_FAILURE,
    PSF_FAILURE:'psf_failure',

    'obj_failure': OBJ_FAILURE,
    OBJ_FAILURE:'obj_failure',
}

def get_flag(val):
    """
    get numerical value for input flag

    Parameters
    ----------
    val: string or int
        string or int form of a flag
    """

    checkflag(val)

    try:
        # first make sure it is numerical
        3 + val
        return val
    except TypeError:
        # it must have been the string version
        return FLAG_MAP[val]

def get_flagname(val):
    """
    get name for input flag

    Parameters
    ----------
    val: string or int
        string or int form of a flag
    """

    checkflag(val)

    try:
        # if it is numerical, return name
        3 + val
        return FLAG_MAP[val]
    except TypeError:
        # it was the string
        return val

def checkflag(val):
    """
    check validity of the input flag

    Parameters
    ----------
    val: string or int
        string or int form of a flag
    """

    assert val in FLAG_MAP,'invalid flag: %s' % val

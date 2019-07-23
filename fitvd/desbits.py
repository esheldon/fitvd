def get_flagvals(flagnames):
    """
    get logical or of all flags in the list of flag names
    """
    flagvals = 0
    for flagname in flagnames:
        flagvals |= get_flagval(flagname)

    return flagvals


def get_flagval(flagname):
    return DESY5_BADPIX_MAP[flagname]


DESY5_BADPIX_MAP = {
    "BPM":          1,  #/* set in bpm (hot/dead pixel/column)        */
    "SATURATE":     2,  #/* saturated pixel                           */
    "INTERP":       4,  #/* interpolated pixel                        */
    "BADAMP":       8,  #/* Data from non-functional amplifier        */
    "CRAY":        16,  #/* cosmic ray pixel                          */
    "STAR":        32,  #/* bright star pixel                         */
    "TRAIL":       64,  #/* bleed trail pixel                         */
    "EDGEBLEED":  128,  #/* edge bleed pixel                          */
    "SSXTALK":    256,  #/* pixel potentially effected by xtalk from  */
                        #/*       a super-saturated source            */
    "EDGE":       512,  #/* pixel flag to exclude CCD glowing edges   */
    "STREAK":    1024,  #/* pixel associated with streak from a       */
                        #/*       satellite, meteor, ufo...           */
    "SUSPECT":   2048,  #/* nominally useful pixel but not perfect    */
    "FIXED":     4096,  #/* corrected by pixcorrect                   */
    "NEAREDGE":  8192,  #/* suspect due to edge proximity             */
    "TAPEBUMP": 16384,  #/* suspect due to known tape bump            */
}

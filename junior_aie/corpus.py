"""Public-only seed notes for the retrieval stack. No private land."""
PUBLIC_NOTES = [
    "Red Feather Lakes Colorado granite boulders. Public forest access varies by season. Check current USFS notices.",
    "Flagstaff Mountain Boulder Colorado. OSMP land. Popular problems. Stay on established approaches.",
    "North Table Mesa Golden Front Range. Jefferson County open space. Crowded weekends.",
    "Mt. Xanadu Red Feather area. Public-access seed pin. Confirm tenure before posting new pins.",
    "JuniorHall is the indoor gym field. gym_internal visibility. Not a public crag.",
    "JuniorStoneField covenant: do not publish private-land boulders without the owner's word of consent.",
    "Offline packs: MBTiles + GPX on device. No vendor guidebook scrape.",
]


def seed(retrieval) -> int:
    for note in PUBLIC_NOTES:
        retrieval.add(note)
    return len(PUBLIC_NOTES)

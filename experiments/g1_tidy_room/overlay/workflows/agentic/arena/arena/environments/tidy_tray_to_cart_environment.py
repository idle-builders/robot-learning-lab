"""The pinned i4h tray task, registered as phase one of the shared room."""

from arena.environments.locomanip_tray_pick_and_place_environment import LocomanipTrayPickAndPlaceEnvironment


class TidyTrayToCartEnvironment(LocomanipTrayPickAndPlaceEnvironment):
    name = "tidy_tray_to_cart"

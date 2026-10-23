"""Safe upstream cart-push baseline; physical tray-to-cart handoff is pending."""

from arena.environments.locomanip_push_cart_environment import LocomanipPushCartEnvironment


class TidyCartToStationEnvironment(LocomanipPushCartEnvironment):
    name = "tidy_cart_to_station"

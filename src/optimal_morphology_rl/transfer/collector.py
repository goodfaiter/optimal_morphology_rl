import asyncio

from AsynchronousPS4Controller import Controller, BaseController
import torch


# monkey patch controller action class to remove print statements
async def _pass(self):
    pass


async def _pass_with_value(self, value):
    pass


BaseController.on_x_press = _pass
BaseController.on_x_release = _pass
BaseController.on_circle_press = _pass
BaseController.on_circle_release = _pass
BaseController.on_triangle_press = _pass
BaseController.on_triangle_release = _pass
BaseController.on_square_press = _pass
BaseController.on_square_release = _pass
BaseController.on_up_arrow_press = _pass
BaseController.on_down_arrow_press = _pass
BaseController.on_up_down_arrow_release = _pass
BaseController.on_left_arrow_press = _pass
BaseController.on_right_arrow_press = _pass
BaseController.on_left_right_arrow_release = _pass
BaseController.on_L1_press = _pass
BaseController.on_L1_release = _pass
BaseController.on_L2_press = _pass_with_value
BaseController.on_L2_release = _pass
BaseController.on_R2_press = _pass_with_value
BaseController.on_R2_release = _pass
BaseController.on_share_press = _pass
BaseController.on_share_release = _pass
BaseController.on_options_press = _pass
BaseController.on_options_release = _pass
BaseController.on_L3_press = _pass
BaseController.on_L3_release = _pass
BaseController.on_R3_press = _pass
BaseController.on_R3_release = _pass
BaseController.on_L3_left = _pass_with_value
BaseController.on_L3_right = _pass_with_value
BaseController.on_L3_up = _pass_with_value
BaseController.on_L3_down = _pass_with_value
BaseController.on_R3_left = _pass_with_value
BaseController.on_R3_right = _pass_with_value
BaseController.on_R3_up = _pass_with_value
BaseController.on_R3_down = _pass_with_value
BaseController.on_R3_x_at_rest = _pass
BaseController.on_R3_y_at_rest = _pass
BaseController.on_L3_x_at_rest = _pass
BaseController.on_L3_y_at_rest = _pass

MAX_STICK_VALUE = 32768.0


class HandController(Controller):

    def __init__(self, path, linear_velocity, angular_velocity, grasp_force):
        Controller.__init__(self, path)
        self.linear_velocity: torch.Tensor = linear_velocity
        self.angular_velocity: torch.Tensor = angular_velocity
        self.grasp_force: torch.Tensor = grasp_force

    # Linear
    async def on_L3_left(self, value):
        self.linear_velocity[1] = -value / MAX_STICK_VALUE

    async def on_L3_right(self, value):
        self.linear_velocity[1] = -value / MAX_STICK_VALUE

    async def on_L3_up(self, value):
        self.linear_velocity[0] = -value / MAX_STICK_VALUE

    async def on_L3_down(self, value):
        self.linear_velocity[0] = -value / MAX_STICK_VALUE

    async def on_up_arrow_press(self):
        self.linear_velocity[2] = 1.0

    async def on_down_arrow_press(self):
        self.linear_velocity[2] = -1.0

    async def on_up_down_arrow_release(self):
        self.linear_velocity[2] = 0.0

    # Angular
    async def on_R3_left(self, value):
        self.angular_velocity[0] = value / MAX_STICK_VALUE

    async def on_R3_right(self, value):
        self.angular_velocity[0] = value / MAX_STICK_VALUE

    async def on_R3_up(self, value):
        self.angular_velocity[1] = -value / MAX_STICK_VALUE

    async def on_R3_down(self, value):
        self.angular_velocity[1] = -value / MAX_STICK_VALUE

    async def on_L1_press(self):
        self.angular_velocity[2] = 1.0

    async def on_L1_release(self):
        self.angular_velocity[2] = 0.0

    async def on_R1_press(self):
        self.angular_velocity[2] = -1.0

    async def on_R1_release(self):
        self.angular_velocity[2] = 0.0

    # grasp
    async def on_R2_press(self, value):
        value += MAX_STICK_VALUE
        value = value / (MAX_STICK_VALUE * 2)
        # current_force = self.grasp_force[0].item()
        # new_force = max(current_force, value)
        self.grasp_force[:] = value
        # print()

    async def on_R2_release(self):
        self.grasp_force[:] = 0.0

    async def on_circle_press(self):
        self.grasp_force[:] = -0.1

    async def on_circle_release(self):
        self.grasp_force[:] = 0.0


async def main():
    controller = HandController(
        path="/dev/input/js0",
        linear_velocity=torch.tensor([0.0, 0.0, 0.0]),
        angular_velocity=torch.tensor([0.0, 0.0, 0.0]),
        grasp_force=torch.tensor([0.0, 0.0]),
    )
    loop = asyncio.get_event_loop()
    loop.create_task(controller.listen(timeout=30))

    while True:
        await asyncio.sleep(0.01)


if __name__ == "__main__":
    asyncio.run(main())

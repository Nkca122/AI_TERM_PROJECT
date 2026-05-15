import pygame
import math
import random as rand
import asyncio


def lerp(A, B, t):
    return A + (B - A) * t


# =========================================================
# LEVEL
# =========================================================
class Level:
    __slots__ = (
        "input_ct",
        "output_ct",
        "input_nodes",
        "output_nodes",
        "biases",
        "weights",
    )

    def __init__(self, input_ct, output_ct):
        self.input_ct = input_ct
        self.output_ct = output_ct

        self.input_nodes = [0.0] * input_ct
        self.output_nodes = [0.0] * output_ct

        self.biases = [0.0] * output_ct
        self.weights = [[0.0 for _ in range(output_ct)] for _ in range(input_ct)]

        self.randomize()

    def randomize(self):
        for i in range(self.input_ct):
            for j in range(self.output_ct):
                self.weights[i][j] = rand.uniform(-1, 1)

        for i in range(self.output_ct):
            self.biases[i] = rand.uniform(-1, 1)

    @staticmethod
    def feed_forward(given_inputs, level):
        for i in range(level.input_ct):
            level.input_nodes[i] = given_inputs[i]

        for i in range(level.output_ct):
            agg = 0.0

            for j in range(level.input_ct):
                agg += level.input_nodes[j] * level.weights[j][i]

            level.output_nodes[i] = math.tanh(agg - level.biases[i])

        return level.output_nodes


# =========================================================
# NN
# =========================================================
class NN:
    __slots__ = ("levels",)

    def __init__(self, neuron_ct):
        self.levels = []

        for i in range(len(neuron_ct) - 1):
            self.levels.append(Level(neuron_ct[i], neuron_ct[i + 1]))

    def clone(self):
        new_net = NN([1])

        new_net.levels = []

        for level in self.levels:
            new_level = Level(level.input_ct, level.output_ct)

            new_level.biases = level.biases[:]
            new_level.weights = [row[:] for row in level.weights]

            new_net.levels.append(new_level)

        return new_net

    @staticmethod
    def feed_forward(given_inputs, network):
        outputs = Level.feed_forward(given_inputs, network.levels[0])

        for i in range(1, len(network.levels)):
            outputs = Level.feed_forward(outputs, network.levels[i])

        return outputs

    @staticmethod
    def mutate(network, amount=1):
        for level in network.levels:

            for i in range(len(level.biases)):
                level.biases[i] = lerp(
                    level.biases[i],
                    rand.uniform(-1, 1),
                    amount,
                )

            for i in range(level.input_ct):
                for j in range(level.output_ct):
                    level.weights[i][j] = lerp(
                        level.weights[i][j],
                        rand.uniform(-1, 1),
                        amount,
                    )


# =========================================================
# CONTROLLER
# =========================================================
class Controller:
    __slots__ = (
        "forward",
        "reverse",
        "left",
        "right",
        "control_type",
    )

    def __init__(self, control_type):
        self.forward = False
        self.reverse = False
        self.left = False
        self.right = False

        self.control_type = control_type

    def _keyboard_listeners(self):
        keys = pygame.key.get_pressed()

        self.forward = keys[pygame.K_w]
        self.reverse = keys[pygame.K_s]
        self.left = keys[pygame.K_a]
        self.right = keys[pygame.K_d]

    def update(self):
        if self.control_type == "KEYS":
            self._keyboard_listeners()

        elif self.control_type == "DUMMY":
            self.forward = True


# =========================================================
# INTERSECTION
# =========================================================
def get_intersection(A, B, C, D):
    tTop = (D.x - C.x) * (A.y - C.y) - (D.y - C.y) * (A.x - C.x)
    uTop = (C.y - A.y) * (A.x - B.x) - (C.x - A.x) * (A.y - B.y)
    bottom = (D.y - C.y) * (B.x - A.x) - (D.x - C.x) * (B.y - A.y)

    if bottom != 0:
        t = tTop / bottom
        u = uTop / bottom

        if 0 <= t <= 1 and 0 <= u <= 1:
            return (
                lerp(A.x, B.x, t),
                lerp(A.y, B.y, t),
                t,
            )

    return None


def polygon_intersect(poly1, poly2):
    for i in range(len(poly1)):
        for j in range(len(poly2)):

            touch = get_intersection(
                poly1[i],
                poly1[(i + 1) % len(poly1)],
                poly2[j],
                poly2[(j + 1) % len(poly2)],
            )

            if touch is not None:
                return True

    return False


# =========================================================
# ROAD
# =========================================================
class Road:
    __slots__ = (
        "x",
        "width",
        "lane_ct",
        "left",
        "right",
        "top",
        "bottom",
        "borders",
    )

    def __init__(self, x, width, lane_ct=3):
        self.x = x
        self.width = width
        self.lane_ct = lane_ct

        self.left = x - width / 2
        self.right = x + width / 2

        self.top = -100000
        self.bottom = 100000

        self.borders = [
            (
                pygame.Vector2(self.left, self.top),
                pygame.Vector2(self.left, self.bottom),
            ),
            (
                pygame.Vector2(self.right, self.top),
                pygame.Vector2(self.right, self.bottom),
            ),
        ]

    def get_lane_center(self, lane_index):
        return (
            self.left
            + (self.width / self.lane_ct) / 2
            + min(self.lane_ct - 1, lane_index) * (self.width / self.lane_ct)
        )

    def draw(self, screen, camera_offset):

        pygame.draw.rect(
            screen,
            (70, 70, 70),
            (
                int(self.left - camera_offset.x),
                -100000,
                int(self.width),
                200000,
            ),
        )

        for count in range(1, self.lane_ct):

            x = lerp(self.left, self.right, count / self.lane_ct)

            pygame.draw.line(
                screen,
                (255, 255, 0),
                (int(x), int(self.top - camera_offset.y)),
                (int(x), int(self.bottom - camera_offset.y)),
                2,
            )

        pygame.draw.line(
            screen,
            (255, 255, 255),
            (int(self.left), int(self.top - camera_offset.y)),
            (int(self.left), int(self.bottom - camera_offset.y)),
            5,
        )

        pygame.draw.line(
            screen,
            (255, 255, 255),
            (int(self.right), int(self.top - camera_offset.y)),
            (int(self.right), int(self.bottom - camera_offset.y)),
            5,
        )


# =========================================================
# SENSOR
# =========================================================
class Sensor:
    __slots__ = (
        "car",
        "ray_ct",
        "ray_length",
        "ray_spread",
        "rays",
        "readings",
    )

    def __init__(self, car, ray_ct=5):

        self.car = car

        self.ray_ct = ray_ct
        self.ray_length = 150
        self.ray_spread = math.pi

        self.rays = []
        self.readings = []

    def _cast_rays(self):

        self.rays.clear()

        base_angle = math.radians(self.car.angle)

        for ray_i in range(self.ray_ct):

            ray_angle = (
                lerp(
                    self.ray_spread / 2,
                    -self.ray_spread / 2,
                    ray_i / (self.ray_ct - 1) if self.ray_ct > 1 else 0.5,
                )
                + base_angle
            )

            sin_a = math.sin(ray_angle)
            cos_a = math.cos(ray_angle)

            start = pygame.Vector2(self.car.pos.x, self.car.pos.y)

            end = pygame.Vector2(
                self.car.pos.x - sin_a * self.ray_length,
                self.car.pos.y - cos_a * self.ray_length,
            )

            self.rays.append((start, end))

    def _get_reading(self, ray, road_borders, traffic):

        closest_touch = None
        min_offset = 999999

        for border in road_borders:

            touch = get_intersection(ray[0], ray[1], border[0], border[1])

            if touch:
                _, _, offset = touch

                if offset < min_offset:
                    min_offset = offset
                    closest_touch = touch

        for traffic_car in traffic:

            # DISTANCE CULLING
            if self.car.pos.distance_squared_to(traffic_car.pos) > 40000:
                continue

            polygon = traffic_car.polygon

            for i in range(len(polygon)):

                touch = get_intersection(
                    ray[0],
                    ray[1],
                    polygon[i],
                    polygon[(i + 1) % len(polygon)],
                )

                if touch:
                    _, _, offset = touch

                    if offset < min_offset:
                        min_offset = offset
                        closest_touch = touch

        return closest_touch

    def update(self, road_borders, traffic):

        self._cast_rays()

        self.readings.clear()

        for ray in self.rays:
            self.readings.append(self._get_reading(ray, road_borders, traffic))

    def draw(self, screen, camera_offset):

        for i, (ray_s, ray_e) in enumerate(self.rays):

            end = ray_e

            if self.readings[i] is not None:

                x, y, _ = self.readings[i]

                end = pygame.Vector2(x, y)

            pygame.draw.line(
                screen,
                (0, 0, 255),
                ray_s - camera_offset,
                end - camera_offset,
                2,
            )

            pygame.draw.line(
                screen,
                (255, 0, 0),
                end - camera_offset,
                ray_e - camera_offset,
                2,
            )


# =========================================================
# CAR
# =========================================================
class Car:
    __slots__ = (
        "pos",
        "width",
        "height",
        "damaged",
        "speed",
        "acceleration",
        "friction",
        "max_speed",
        "max_reverse_speed",
        "angle",
        "turn_speed",
        "polygon",
        "controller",
        "automatic",
        "sensor",
        "NN",
    )

    def __init__(
        self,
        pos,
        width,
        height,
        control_type,
        max_speed,
    ):

        self.pos = pygame.Vector2(pos)

        self.width = width
        self.height = height

        self.damaged = False

        self.speed = 0

        self.acceleration = 200
        self.friction = 120

        self.max_speed = max_speed
        self.max_reverse_speed = -max_speed

        self.angle = 0
        self.turn_speed = 120

        self.polygon = []

        self.controller = Controller(control_type)

        self.automatic = control_type == "AI"

        self.sensor = Sensor(self) if control_type != "DUMMY" else None

        self.NN = NN([5, 4, 4]) if control_type != "DUMMY" else None

        self.polygon = self._create_polygon()

    def _assess_damage(self, road_borders, traffic):

        for border in road_borders:
            if polygon_intersect(self.polygon, border):
                return True

        for traffic_car in traffic:

            if polygon_intersect(
                self.polygon,
                traffic_car.polygon,
            ):
                return True

        return False

    def _create_polygon(self):

        points = []

        hw = self.width / 2
        hh = self.height / 2

        rad = math.radians(self.angle)

        sin_a = math.sin(rad)
        cos_a = math.cos(rad)

        corners = [
            (-hw, -hh),
            (hw, -hh),
            (hw, hh),
            (-hw, hh),
        ]

        for x, y in corners:

            rx = x * cos_a - y * sin_a
            ry = -(x * sin_a + y * cos_a)

            points.append(
                pygame.Vector2(
                    self.pos.x + rx,
                    self.pos.y + ry,
                )
            )

        return points

    def _move(self, dt):

        if self.controller.forward:
            self.speed += self.acceleration * dt

        elif self.controller.reverse:
            self.speed -= self.acceleration * dt

        else:

            if self.speed > 0:
                self.speed = max(
                    0,
                    self.speed - self.friction * dt,
                )

            elif self.speed < 0:
                self.speed = min(
                    0,
                    self.speed + self.friction * dt,
                )

        self.speed = max(
            self.max_reverse_speed,
            min(self.speed, self.max_speed),
        )

        if self.speed != 0:

            direction = 1 if self.speed > 0 else -1

            if self.controller.left:
                self.angle += self.turn_speed * dt * direction

            if self.controller.right:
                self.angle -= self.turn_speed * dt * direction

        rad = math.radians(self.angle)

        sin_a = math.sin(rad)
        cos_a = math.cos(rad)

        self.pos.x -= sin_a * self.speed * dt
        self.pos.y -= cos_a * self.speed * dt

    def update(self, dt, road_borders, traffic):

        if self.damaged:
            return

        self.controller.update()

        self._move(dt)

        self.polygon = self._create_polygon()

        self.damaged = self._assess_damage(
            road_borders,
            traffic,
        )

        if self.sensor is not None:

            self.sensor.update(
                road_borders,
                traffic,
            )

            offsets = []

            for reading in self.sensor.readings:

                if reading:
                    offsets.append(1 - reading[2])
                else:
                    offsets.append(0)

            outputs = NN.feed_forward(
                offsets,
                self.NN,
            )

            if self.automatic:

                self.controller.forward = outputs[0] > 0
                self.controller.left = outputs[1] > 0
                self.controller.right = outputs[2] > 0
                self.controller.reverse = outputs[3] > 0

    def draw(self, screen, camera_offset, draw_sensor=False):

        pygame.draw.polygon(
            screen,
            (
                (0, 255, 0)
                if self.damaged
                else ((0, 0, 0) if self.automatic else (255, 0, 0))
            ),
            [
                (
                    int(point.x - camera_offset.x),
                    int(point.y - camera_offset.y),
                )
                for point in self.polygon
            ],
        )

        if self.sensor is not None and draw_sensor:
            self.sensor.draw(screen, camera_offset)


# =========================================================
# MAIN
# =========================================================
async def main():

    pygame.init()
    pygame.font.init()

    font = pygame.font.Font(
        "freesansbold.ttf",
        32,
    )

    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    road = Road(screen.get_width() / 2, 400)

    N = 10

    mutate_amount = 0.5
    generation_timer = 0
    GEN_LIMIT = 15

    gen = 0

    def generate_cars(count):

        return [
            Car(
                (
                    road.get_lane_center(1),
                    screen.get_height() / 2,
                ),
                30,
                40,
                "AI",
                500,
            )
            for _ in range(count)
        ]

    def generate_traffic():

        return [
            Car(
                (road.get_lane_center(0), screen.get_height() / 2 - 100),
                30,
                40,
                "DUMMY",
                300,
            ),
            Car(
                (road.get_lane_center(1), screen.get_height() / 2 - 80),
                30,
                40,
                "DUMMY",
                400,
            ),
            Car(
                (road.get_lane_center(2), screen.get_height() / 2 - 120),
                30,
                40,
                "DUMMY",
                200,
            ),
            Car(
                (road.get_lane_center(2), screen.get_height() / 2 - 200),
                30,
                40,
                "DUMMY",
                200,
            ),
            Car(
                (road.get_lane_center(1), screen.get_height() / 2 - 180),
                30,
                40,
                "DUMMY",
                450,
            ),
            Car(
                (road.get_lane_center(0), screen.get_height() / 2 - 220),
                30,
                40,
                "DUMMY",
                300,
            ),
        ]

    cars = generate_cars(N)

    best_nn = cars[0].NN.clone()

    for i in range(N):

        cars[i].NN = best_nn.clone()

        if i != 0:
            NN.mutate(cars[i].NN, mutate_amount)

    traffic = generate_traffic()

    camera_offset = pygame.Vector2(0, 0)

    running = True

    while running:

        dt = clock.tick(60) / 1000

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 120, 0))

        # ==========================================
        # UPDATE TRAFFIC
        # ==========================================
        for traffic_car in traffic:
            traffic_car.update(
                dt,
                road.borders,
                [],
            )

        # ==========================================
        # UPDATE AI
        # ==========================================
        active_cars = []

        for car in cars:

            car.update(
                dt,
                road.borders,
                traffic,
            )

            if not car.damaged:
                active_cars.append(car)

        best_car = min(
            cars,
            key=lambda c: c.pos.y,
        )

        best_active_car = (
            min(active_cars, key=lambda c: c.pos.y) if active_cars else best_car
        )

        # ==========================================
        # NEXT GENERATION
        # ==========================================
        generation_timer += dt

        if generation_timer >= GEN_LIMIT or not active_cars:

            generation_timer = 0

            gen += 1

            best_nn = best_car.NN.clone()

            cars = generate_cars(N)

            for i in range(N):

                cars[i].NN = best_nn.clone()

                if i != 0:
                    NN.mutate(
                        cars[i].NN,
                        mutate_amount,
                    )

            traffic = generate_traffic()

        # ==========================================
        # CAMERA
        # ==========================================
        camera_offset.y = best_active_car.pos.y - screen.get_height() / 2

        # ==========================================
        # DRAW
        # ==========================================
        road.draw(screen, camera_offset)

        for traffic_car in traffic:
            traffic_car.draw(
                screen,
                camera_offset,
            )

        best_active_car.draw(
            screen,
            camera_offset,
            True,
        )

        text = font.render(
            f"GEN : {gen}",
            True,
            (255, 255, 255),
        )

        screen.blit(text, (20, 20))

        pygame.display.update()

        await asyncio.sleep(0)

    pygame.quit()


asyncio.run(main())

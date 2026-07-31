from ursina import *
import math

class Body(Entity):
    def __init__(self, position=(0,0,0), velocity=(0,0,0), mass=1, radius=1, color=color.white):
        # Initialiseer de Ursina Entity (3D vorm)
        super().__init__(
            model='sphere',       
            color=color,
            scale=radius * 2,     
            position=position
        )
        self.vx, self.vy, self.vz = velocity
        self.mass = mass
        
        # Eigen 3D trail constructie met een Mesh (Lijn)
        self.trail_points = []
        self.max_trail = 120
        self.trail_entity = Entity(
            model=Mesh(vertices=[], mode='line', thickness=3), 
            color=color
        )

    def attract(self, other):
        # 3D afstandsberekening
        dx = other.x - self.x
        dy = other.y - self.y
        dz = other.z - self.z
        distance = math.sqrt(dx**2 + dy**2 + dz**2)

        if distance < 1: 
            return

        G = 1 
        force = G * (self.mass * other.mass) / (distance**2)

        fx = force * (dx / distance)
        fy = force * (dy / distance)
        fz = force * (dz / distance)

        self.vx += fx / self.mass
        self.vy += fy / self.mass
        self.vz += fz / self.mass

    def move(self):
        # Update positie op basis van snelheid en de framerate (time.dt)
        self.x += self.vx * time.dt
        self.y += self.vy * time.dt
        self.z += self.vz * time.dt
        
        # Voeg huidige positie toe aan het spoor
        self.trail_points.append(Vec3(self.x, self.y, self.z))
        
        if len(self.trail_points) > self.max_trail:
            self.trail_points.pop(0)
            
        # Genereer de mesh pas als we minimaal 2 punten hebben om een lijn te trekken
        if len(self.trail_points) >= 2:
            self.trail_entity.model.vertices = self.trail_points
            self.trail_entity.model.generate()
import pygame
import math
import pandas as pd

class EnvCircle:
    def __init__(self, w=600, h=400):
        self.traj = [[0,0,0]]
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Circle')
        self.clock = pygame.time.Clock()
        self.next_tick = 500
        self.speed = 40
        self.reset()
        self.done = False

    def step(self,angle, radius, coords):
        theta = math.radians(angle)
        return [coords[0] + radius * math.cos(theta), coords[1] + radius * math.sin(theta)]

    def is_done(self,angle):
        if angle==360 or angle ==-360:
            return True
        return False

    def reset(self):         
        self.coords = 300, 200
        self.angle = 0
        self.rect = pygame.Rect(*self.coords,20,20)
        self.update_ui()

    def update_ui(self):
        self.display.fill((0,0,30))
        self.display.fill((0,150,0), self.rect)
        pygame.display.flip()
        self.clock.tick(self.speed)

    def draw(self):
        game_over = False
        num_step = 0 
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            ticks = pygame.time.get_ticks() 
            if ticks > self.next_tick:
                self.next_tick += self.speed
                self.angle += 1
                self.radius = 1
                self.coords = self.step(self.angle, self.radius, self.coords)
                self.traj.append([self.angle,self.radius,2])
                self.rect.topleft = self.coords
                game_over = self.is_done(self.angle)

                num_step +=1
                
            self.update_ui()

            if game_over:
                break
        pygame.quit()

 
if __name__ == '__main__':
    env = EnvCircle()
    env.draw()
            
    traj = pd.DataFrame(env.traj)
    traj.to_csv('traj.csv',index=False)

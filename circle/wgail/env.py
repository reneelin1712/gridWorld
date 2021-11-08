import pygame
import math
import pandas as pd

pygame.init()
font = pygame.font.Font(None, 35)

WHITE = (255, 255, 255)
SPEED = 300
class EnvCircle:
    def __init__(self, w=600, h=400):
        self.traj = []
        self.w = w
        self.h = h
        self.done = False
        self.update_n = 0
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Circle')
        self.clock = pygame.time.Clock()
        self.reset()
    

    def to_loc(self,angle, radius, coords):
        theta = math.radians(angle)
        # print('theta',math.cos(theta))
        # print('x',radius * math.cos(theta))
        return [coords[0] + radius * math.cos(theta), coords[1] + radius * math.sin(theta)]
        
    def step(self,action):
        table = [[0,0],[0,1],[1,0],[1,1]]
        angle_idx,radius_idx = table[action]
        angle = [-1,1][angle_idx]
        radius = [1,2][radius_idx]
        # print('each step action', angle)
        self.angle = self.angle + angle
        # print('self angle',self.angle)
        
        self.radius = radius
        next_state = [self.angle, self.radius]

        reward = 0
        done = self.is_done(self.angle)
        info = {}

        self.move()
        self.update_ui()
        self.clock.tick(SPEED)
        return next_state,reward,done,info

    def is_done(self,angle):
        if angle==360 or angle ==-360:
            return True
        return False

    def reset(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False         
        self.coords = 300, 200
        self.angle = 0
        self.rect = pygame.Rect(*self.coords,20,20)
        self.update_ui()

    def update_ui(self):
        self.display.fill((0,0,30))
        self.display.fill((0,150,0), self.rect)
        text = font.render("Episode: " + str(self.update_n), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        self.clock.tick(SPEED)

    def move(self):
        # game_over = False
        # num_step = 0 
        # while True:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             running = False
            
        #     ticks = pygame.time.get_ticks() 
        #     if ticks > self.next_tick:
        # self.next_tick += self.speed
        # self.angle -= 1
        # self.radius = 1
        self.coords = self.to_loc(self.angle, self.radius, self.coords)
        # print(self.coords)
        # self.traj.append([self.angle,self.radius])
        self.rect.topleft = self.coords
        # print('coord',self.coords)
        #         game_over = self.is_done(self.coords)

        #         num_step +=1
                
        #     self.update_ui()

        #     if game_over:
        #         break
        # pygame.quit()


    def update_num(self,update_i):
        self.update_n = update_i




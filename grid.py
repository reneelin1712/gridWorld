import pygame

pygame.init()

class GridWorld:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Grid World')
        self.clock = pygame.time.Clock()
        
        # init game state
        self.agent = Point(self.w/2, self.h/2)
        self.target = 
        self.score = 0

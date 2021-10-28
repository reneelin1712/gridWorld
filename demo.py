import pygame

pygame.init()
font = pygame.font.Font(None, 20)

### UI ###
# screen size
w = 320
h = 240

# tile size
BLOCK_SIZE = 40

# color
WHITE = (255, 255, 255)
BLACK = (0,0,0)
RED = (200,0,0)
BLUE = (0, 0, 255)


# setup screen
screen = pygame.display.set_mode((w, h))
pygame.display.set_caption('Grid World')

# setup timer
clock = pygame.time.Clock()
# tick 
SPEED = 60

# agent position
x = 0
y = h-BLOCK_SIZE

# game state
off_screen = False
arrived = False
score = 0
    
### Game Loop/Progress
# Run until the player quit
playing = True
while playing:

    # Player events: mouse click, keyboard
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            playing = False

        # get the player action
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                x -= BLOCK_SIZE
            elif event.key == pygame.K_RIGHT:
                x += BLOCK_SIZE
            elif event.key == pygame.K_UP:
                y -= BLOCK_SIZE
            elif event.key == pygame.K_DOWN:
                y += BLOCK_SIZE

        # move off the screen
        if x < 0 or x > w-BLOCK_SIZE or y<0 or y>h-BLOCK_SIZE:
            off_screen = True

        # arrives the destination
        if x== w-BLOCK_SIZE and y ==0:
            arrived = True
            score = 100

            
    screen.fill(BLACK)        
    # Draw a agent in the left bottom, a target on the top right
    pygame.draw.rect(screen, RED, pygame.Rect(x,y, BLOCK_SIZE, BLOCK_SIZE))
    pygame.draw.rect(screen, BLUE, pygame.Rect(w-BLOCK_SIZE, 0, BLOCK_SIZE, BLOCK_SIZE))

    # display the score
    text = font.render("Score: " + str(score), True, WHITE)
    screen.blit(text, [0, 0])

    # Flip the display
    pygame.display.flip()
    clock.tick(SPEED)

    if off_screen == True or arrived ==True:
        break

pygame.quit()
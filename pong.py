import random
import pygame, sys
from pygame.locals import *
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
import threading


pygame.init()
fps = pygame.time.Clock()
load_dotenv(override=True)

#colors
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLACK = (0,0,0)

#globals
WIDTH = 600
HEIGHT = 400       
BALL_RADIUS = 20
PAD_WIDTH = 8
PAD_HEIGHT = 80
HALF_PAD_WIDTH = PAD_WIDTH // 2
HALF_PAD_HEIGHT = PAD_HEIGHT // 2
ball_pos = [0,0]
ball_vel = [0,0]
paddle1_vel = 0
paddle2_vel = 0
l_score = 0
r_score = 0

#canvas declaration
window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption('Pong')

# helper function that spawns a ball, returns a position vector and a velocity vector
# if right is True, spawn to the right, else spawn to the left
def ball_init(right):
    global ball_pos, ball_vel # these are vectors stored as lists
    ball_pos = [WIDTH//2,HEIGHT//2]
    horz = random.randrange(2,4)
    vert = random.randrange(1,3)
    
    if right == False:
        horz = - horz
        
    ball_vel = [horz,-vert]

# define event handlers
def init():
    global paddle1_pos, paddle2_pos, paddle1_vel, paddle2_vel,l_score,r_score  # these are floats
    global score1, score2  # these are ints
    paddle1_pos = [HALF_PAD_WIDTH - 1,HEIGHT//2]
    paddle2_pos = [WIDTH +1 - HALF_PAD_WIDTH,HEIGHT//2]
    l_score = 0
    r_score = 0
    if random.randrange(0,2) == 0:
        ball_init(True)
    else:
        ball_init(False)


#draw function of canvas
def draw(canvas):
    global paddle1_pos, paddle2_pos, ball_pos, ball_vel, l_score, r_score
    canvas.fill(BLACK)
    pygame.draw.line(canvas, WHITE, [WIDTH // 2, 0],[WIDTH // 2, HEIGHT], 1)
    pygame.draw.line(canvas, WHITE, [PAD_WIDTH, 0],[PAD_WIDTH, HEIGHT], 1)
    pygame.draw.line(canvas, WHITE, [WIDTH - PAD_WIDTH, 0],[WIDTH - PAD_WIDTH, HEIGHT], 1)
    pygame.draw.circle(canvas, WHITE, [WIDTH//2, HEIGHT//2], 70, 1)

    # update paddle's vertical position, keep paddle on the screen
    if paddle1_pos[1] > HALF_PAD_HEIGHT and paddle1_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
        paddle1_pos[1] += paddle1_vel
    elif paddle1_pos[1] == HALF_PAD_HEIGHT and paddle1_vel > 0:
        paddle1_pos[1] += paddle1_vel
    elif paddle1_pos[1] == HEIGHT - HALF_PAD_HEIGHT and paddle1_vel < 0:
        paddle1_pos[1] += paddle1_vel
    
    if paddle2_pos[1] > HALF_PAD_HEIGHT and paddle2_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
        paddle2_pos[1] += paddle2_vel
    elif paddle2_pos[1] == HALF_PAD_HEIGHT and paddle2_vel > 0:
        paddle2_pos[1] += paddle2_vel
    elif paddle2_pos[1] == HEIGHT - HALF_PAD_HEIGHT and paddle2_vel < 0:
        paddle2_pos[1] += paddle2_vel

    #update ball
    ball_pos[0] += int(ball_vel[0])
    ball_pos[1] += int(ball_vel[1])

    #draw paddles and ball
    pygame.draw.circle(canvas, RED, ball_pos, 20, 0)
    pygame.draw.polygon(canvas, GREEN, [[paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT], [paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT], [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT], [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT]], 0)
    pygame.draw.polygon(canvas, GREEN, [[paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT], [paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT], [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT], [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT]], 0)

    #ball collision check on top and bottom walls
    if int(ball_pos[1]) <= BALL_RADIUS:
        ball_vel[1] = - ball_vel[1]
    if int(ball_pos[1]) >= HEIGHT + 1 - BALL_RADIUS:
        ball_vel[1] = -ball_vel[1]
    
    #ball collison check on gutters or paddles
    if int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH and int(ball_pos[1]) in range(paddle1_pos[1] - HALF_PAD_HEIGHT,paddle1_pos[1] + HALF_PAD_HEIGHT,1):
        ball_vel[0] = -ball_vel[0]
        ball_vel[0] *= 1.1
        ball_vel[1] *= 1.1
    elif int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH:
        r_score += 1
        ball_init(True)
        
    if int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH and int(ball_pos[1]) in range(paddle2_pos[1] - HALF_PAD_HEIGHT,paddle2_pos[1] + HALF_PAD_HEIGHT,1):
        ball_vel[0] = -ball_vel[0]
        ball_vel[0] *= 1.1
        ball_vel[1] *= 1.1
    elif int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH:
        l_score += 1
        ball_init(False)

    #update scores
    myfont1 = pygame.font.SysFont("Comic Sans MS", 20)
    label1 = myfont1.render("Score "+str(l_score), 1, (255,255,0))
    canvas.blit(label1, (50,20))

    myfont2 = pygame.font.SysFont("Comic Sans MS", 20)
    label2 = myfont2.render("Score "+str(r_score), 1, (255,255,0))
    canvas.blit(label2, (470, 20))  
    
    
#keydown handler
def keydown(event):
    global paddle2_vel
    
    if event.key == K_UP:
        paddle2_vel = -8
    elif event.key == K_DOWN:
        paddle2_vel = 8

#keyup handler
def keyup(event):
    global paddle2_vel
    
    if event.key in (K_UP, K_DOWN):
        paddle2_vel = 0

init()

@tool
def move_up() -> None:
    """
    This tool moves the paddle up
    Returns: 
        None
    """
    global paddle1_vel
    print("moving the paddle up...")
    paddle1_vel = -8

@tool
def move_down() -> None:
    """
    This tool moves the paddle down
    Returns: 
        None
    """
    global paddle1_vel
    print("moving the paddle down...")
    paddle1_vel = 8

@tool
def stop_paddle() -> None:
    """
    This tool stops paddle movement
    Returns: 
        None
    """
    global paddle1_vel
    print("Stopping paddle...")
    paddle1_vel = 0

@tool
def get_ball_position() -> list:
    """
    This tool gets the position of the ball
    Returns: 
        list: The position of the ball
    """
    global ball_pos
    print("Getting ball position...")
    print(f"Ball position{ball_pos[1]}")
    return ball_pos[1]

@tool
def get_paddle_position() -> list:
    """
    This tool gets the position of the paddle
    Returns: 
        list: The position of the paddle
    """
    global paddle1_pos
    print(f"Paddle position{paddle1_pos[1]}")
    return paddle1_pos[1]

def pong_agent():
    """Create a React agent with the specified tools."""
    chat = ChatOpenAI(temperature=0, model="gpt-4o")
    print("pong agent running...")

    system_prompt = f"""
    You are an agent that can use the following tools to play a game of pong:
    - move_up: This tool moves the paddle up
    - move_down: This tool moves the paddle down
    - stop_paddle: This tool stops paddle movement
    - get_ball_position: This tool gets the position of the ball
    - get_paddle_position: This tool gets the position of the paddle

    Your task is to determine the best move for the paddle, based on the ball position.
    The following are rules you must follow in order to find the best move for the paddle:
        - If the value returned from get_ball_position is lower than the value returned from get_paddle_position move up using the move_up tool.
        - If the value returned from get_ball_position is higher than the value returned from get_paddle_position move down using the move_down tool.
        - After a brief amount of time, (0.25 seconds) stop the paddle using the stop_paddle tool.
    """

    tools = [
        move_up,
        move_down,
        stop_paddle,
        get_ball_position,
        get_paddle_position
    ]

    return create_react_agent(
        model=chat,
        tools=tools,
        prompt=system_prompt,
        checkpointer=MemorySaver(),
    )


if __name__ == "__main__":
    agent = pong_agent()

    config = {
        "configurable": {
            "thread_id": 42,
            "recursion_limit": 25
        }
    }  

    # asynchronously invoke the agent
    def invoke_agent():
        """Function to invoke the agent asynchronously in a separate thread."""
        agent.invoke({"messages": ["start"]}, config=config)

    invoke_agent_counter = 0

while True:
    draw(window)
    if invoke_agent_counter % 10 == 0:
        agent_thread = threading.Thread(target=invoke_agent, daemon=True)
        agent_thread.start()
        
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            keydown(event)
        elif event.type == KEYUP:
            keyup(event)
        elif event.type == QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    invoke_agent_counter += 1
    fps.tick(30) 
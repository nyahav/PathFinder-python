import pygame
import os
import math
from queue import PriorityQueue
from collections import deque

from rich import _console

os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'
pygame.init()
pygame.font.init() 

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH + 100), pygame.RESIZABLE)  # Increased height to fit the button
pygame.display.set_caption("Path Finding Visualizer")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)
BUTTON_COLOR = (0, 0, 255)

MAZE_PRESETS = {
    "Maze 1": [
        (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), 
        (4, 5), (4, 6), (5, 6), (6, 6), (6, 7), (6, 8), (6, 9), (7, 9), 
        (8, 9), (9, 9), (10, 9), (10, 8), (10, 7), (11, 7), (12, 7), 
        (12, 6), (13, 6), (14, 6), (15, 6), (16, 6), (16, 7), (16, 8), 
        (16, 9), (16, 10), (15, 10), (15, 11), (15, 12), (14, 12), 
        (14, 13), (14, 14), (14, 15), (13, 15), (12, 15), (12, 14), 
        (12, 13), (11, 13), (11, 12), (11, 11), (10, 11), (9, 11), 
        (9, 10), (9, 12), (8, 12), (7, 12), (6, 12), (6, 13), (5, 13), 
        (5, 14), (5, 15), (4, 15), (4, 16), (4, 17), (3, 17), (3, 18), 
        (3, 19), (4, 19), (5, 19), (5, 20), (5, 21), (6, 21), (6, 22), 
        (7, 22), (8, 22), (8, 23), (8, 24), (9, 24), (10, 24), (11, 24), 
        (12, 24), (12, 25), (12, 26), (11, 26), (10, 26), (9, 26), 
        (9, 27), (9, 28), (8, 28), (7, 28), (6, 28), (5, 28), (4, 28), 
        (3, 28), (2, 28), (1, 28), (0, 28), (0, 29), (0, 30), (0, 31), 
        (1, 31), (2, 31), (2, 32), (3, 32), (4, 32), (5, 32), (6, 32), 
        (7, 32), (8, 32), (9, 32), (10, 32), (11, 32), (12, 32), (13, 32), 
        (14, 32), (15, 32), (16, 32), (17, 32), (18, 32), (19, 32), 
        (20, 32), (21, 32), (22, 32), (23, 32), (24, 32), (25, 32), 
        (26, 32), (27, 32), (28, 32), (29, 32), (30, 32), (31, 32), 
        (32, 32), (33, 32), (34, 32), (35, 32), (36, 32), (37, 32), 
        (38, 32), (39, 32), (40, 32), (41, 32), (42, 32), (43, 32), 
        (44, 32), (45, 32), (46, 32), (47, 32), (48, 32), (49, 32), 
        (49, 33), (49, 34), (49, 35), (49, 36), (49, 37), (49, 38), 
        (49, 39), (49, 40), (49, 41), (49, 42), (49, 43), (49, 44), 
        (49, 45), (49, 46), (49, 47), (49, 48), (49, 49)
    ],
    "Maze 2": [
        (0, 1), (0, 2), (0, 3), (1, 3), (1, 4), (1, 5), (2, 5), (3, 5), 
        (3, 6), (4, 6), (5, 6), (5, 7), (5, 8), (6, 8), (7, 8), (8, 8), 
        (9, 8), (10, 8), (11, 8), (11, 9), (11, 10), (11, 11), (12, 11), 
        (13, 11), (13, 12), (14, 12), (14, 13), (15, 13), (15, 14), 
        (15, 15), (16, 15), (16, 16), (16, 17), (16, 18), (17, 18), 
        (18, 18), (18, 19), (19, 19), (20, 19), (20, 20), (21, 20), 
        (21, 21), (21, 22), (22, 22), (23, 22), (24, 22), (25, 22), 
        (26, 22), (27, 22), (28, 22), (28, 23), (28, 24), (28, 25), 
        (28, 26), (29, 26), (30, 26), (31, 26), (32, 26), (33, 26), 
        (34, 26), (35, 26), (35, 27), (35, 28), (35, 29), (36, 29), 
        (37, 29), (38, 29), (39, 29), (40, 29), (40, 30), (41, 30), 
        (42, 30), (42, 31), (42, 32), (43, 32), (43, 33), (43, 34), 
        (44, 34), (44, 35), (44, 36), (44, 37), (44, 38), (44, 39), 
        (44, 40), (44, 41), (44, 42), (44, 43), (44, 44), (44, 45), 
        (44, 46), (44, 47), (44, 48), (44, 49), (45, 49), (46, 49), 
        (47, 49), (48, 49), (49, 49)
    ],
}

class Spot:
	def __init__(self, row, col, width, total_rows):
		self.row = row
		self.col = col
		self.x = row * width
		self.y = col * width
		self.color = WHITE
		self.neighbors = []
		self.width = width
		self.total_rows = total_rows

	def get_pos(self):
		return self.row, self.col

	def is_closed(self):
		return self.color == RED

	def is_open(self):
		return self.color == GREEN

	def is_barrier(self):
		return self.color == BLACK

	def is_start(self):
		return self.color == ORANGE

	def is_end(self):
		return self.color == TURQUOISE

	def reset(self):
		self.color = WHITE

	def make_start(self):
		self.color = ORANGE

	def make_closed(self):
		self.color = RED

	def make_open(self):
		self.color = GREEN

	def make_barrier(self):
		self.color = BLACK

	def make_end(self):
		self.color = TURQUOISE

	def make_path(self):
		self.color = PURPLE

	def draw(self, win):
		pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

	def update_neighbors(self, grid):
		self.neighbors = []
        # DOWN
		if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): 
			self.neighbors.append(grid[self.row + 1][self.col])
        # UP
		if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): 
			self.neighbors.append(grid[self.row - 1][self.col])
        # RIGHT
		if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): 
			self.neighbors.append(grid[self.row][self.col + 1])
        # LEFT
		if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): 
			self.neighbors.append(grid[self.row][self.col - 1])
         
         
            
	def __lt__(self, other):
		return False
def reconstruct_path(came_from,current,draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()
    
def algorithmBFS(draw, grid, start, end):
    queue = deque([start])
    came_from = {}
    
    
    visited = {spot: False for row in grid for spot in row}
    visited[start] = True

    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = queue.popleft()  

        
        if current == end:
            reconstruct_path(came_from, end, draw)
            start.make_start()
            end.make_end()
            return True

        for neighbor in current.neighbors:
            if not visited[neighbor] and not neighbor.is_barrier():
                came_from[neighbor] = current
                visited[neighbor] = True
                queue.append(neighbor)  
                if neighbor != start:
                 neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return None  

def algorithmDFS(draw, grid, start, end):
    
    stack = [start]
    came_from = {}
    
    
    visited = {spot: False for row in grid for spot in row}
    visited[start] = True

    while stack:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = stack.pop()  

       
        if current == end:
            reconstruct_path(came_from, end, draw)
            start.make_start()
            end.make_end()
            return True

        for neighbor in current.neighbors:
            if not visited[neighbor] and not neighbor.is_barrier():
                came_from[neighbor] = current
                visited[neighbor] = True
                stack.append(neighbor) 
                if neighbor != start:
                 neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return None

def algorithmDijkstra(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()  
    open_set.put((0, count, start))  
    came_from = {}  
    
    g_score = {spot: float("inf") for row in grid for spot in row}  
    g_score[start] = 0  
    
    open_set_hash = {start}  

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        current = open_set.get()[2]  
        open_set_hash.remove(current)

        if current == end:  
            reconstruct_path(came_from, end, draw)
            start.make_start()
            end.make_end()
            return True

        for neighbor in current.neighbors: 
            temp_g_score = g_score[current] + 1  

            if temp_g_score < g_score[neighbor]:  
                came_from[neighbor] = current  
                g_score[neighbor] = temp_g_score  
                
                if neighbor not in open_set_hash:  
                    count += 1
                    open_set.put((g_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()  
        
        draw()  
        
        if current != start:
            current.make_closed()  

    return False  

def algorithmAstart(draw,grid,start,end):
    count = 0
    open_set= PriorityQueue()
    open_set.put((0,count,start))
    came_from={}
    g_score= {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score= {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(),end.get_pos())
    
    #has all the spots as the priorityQueue,but more effiecent for searching if spot is in there
    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                
        current = open_set.get()[2]
        #sync between queue and hash
        open_set_hash.remove(current)
        
        if current == end:
            reconstruct_path(came_from,end,draw)
            start.make_start()
            end.make_end()
            return True
        
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
        
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(),end.get_pos())
                if neighbor not in open_set_hash:
                    count +=1
                    open_set.put((f_score[neighbor],count,neighbor))
                    open_set_hash.add(neighbor)
                    if neighbor != start:
                     neighbor.make_open()
        
        draw()
        
        if current != start:
            current.make_closed()
    return None   
    
def h(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	return abs(x1 - x2) + abs(y1 - y2)

def make_grid(rows, width):
    grid = []
    gap = width // rows
    
    # Add top border (row of black spots)
    top_border = [Spot(0, j, gap, rows + 2) for j in range(rows + 2)]
    for spot in top_border:
        spot.make_barrier()  # Make these spots black
    grid.append(top_border)
    
    # Create the grid, wrapped with black barrier spots on left and right
    for i in range(1, rows + 1):  # Shift all rows by 1
        row = []
        
        # Left border spot
        left_border_spot = Spot(i, 0, gap, rows + 2)
        left_border_spot.make_barrier()
        row.append(left_border_spot)

        # Add the actual spots from the grid
        for j in range(1, rows + 1):
            spot = Spot(i, j, gap, rows + 2)
            row.append(spot)
        
        # Right border spot
        right_border_spot = Spot(i, rows + 1, gap, rows + 2)
        right_border_spot.make_barrier()
        row.append(right_border_spot)
        
        grid.append(row)
    
    # Add bottom border (row of black spots)
    bottom_border = [Spot(rows + 1, j, gap, rows + 2) for j in range(rows + 2)]
    for spot in bottom_border:
        spot.make_barrier()
    grid.append(bottom_border)

    return grid
'''
def make_grid(rows, width):
	grid = []
	gap = width // rows
	for i in range(rows):
		grid.append([])
		for j in range(rows):
			spot = Spot(i, j, gap, rows)
			grid[i].append(spot)

	return grid
'''

def draw_grid(win, rows, width):
	gap = width // rows
	for i in range(rows):
		pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
		for j in range(rows):
			pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))
   
def generate_maze(width, height):
    maze = []
    for i in range(width):
        maze.append([0] * height)

    def backtrack(x, y):
        maze[x][y] = 1
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and maze[nx][ny] == 0:
                backtrack(nx, ny)
                maze[x + dx // 2][y + dy // 2] = 1

    backtrack(0, 0)
    return maze

def maze_to_string(maze):
    maze_str = "Maze 2: [\n"
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 1:
                maze_str += f"    ({i}, {j}),\n"
    maze_str += "]"
    return maze_str
def load_maze(grid, maze_name):
    # Clear existing barriers before loading new maze
    for row in grid:
        for spot in row:
            if not spot.is_start() and not spot.is_end():
                spot.reset()

    if maze_name == "random":
        # Generate a random maze using the generate_maze function
        maze = generate_maze(50, 50)
        maze_string = maze_to_string(maze)
        _console.log(maze_string)

        # Load the generated maze into the grid
        for (row, col) in maze_string:
            grid[row][col].make_barrier()

    # Load the selected maze preset if it's not "random"
    elif maze_name in MAZE_PRESETS:
        for (row, col) in MAZE_PRESETS[maze_name]:
            grid[row][col].make_barrier()
            
def draw_button(win, x, y, w, h, text, color):
    border_radius = 10
    pygame.draw.rect(win, BLACK, (x - 5, y - 5, w + 10, h + 10), border_radius=border_radius) 
    pygame.draw.rect(win, color, (x, y, w, h), border_radius=border_radius) 
    font = pygame.font.Font(None, 40)
    text_surface = font.render(text, True, WHITE)
    win.blit(text_surface, (x + 10, y + 10))  

def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width)

    
    button_width, button_height = 100, 50
    button_x_restart = width // 2 - 120
    button_x_start = width // 2 + 20
    button_y = width + 20

    draw_button(win, button_x_restart, button_y, button_width, button_height, "Restart", BUTTON_COLOR)
    draw_button(win, button_x_start, button_y, button_width, button_height, "Start", BUTTON_COLOR)

    pygame.display.update()


def is_button_clicked(pos, button_x, button_y, button_width, button_height):
    x, y = pos
    return button_x <= x <= button_x + button_width and button_y <= y <= button_y + button_height

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    
    if row >= rows or col >= rows or row < 0 or col < 0:
        raise IndexError("Clicked position out of grid bounds")  

    return row, col

class Dropdown:
    def __init__(self, x, y, width, options):
        self.rect = pygame.Rect(x, y, width, 40)
        self.options = options
        self.selected = options[0]
        self.dropdown_open = False
        self.font = pygame.font.Font(None, 30)

    def draw(self, win):
        pygame.draw.rect(win, BLUE, self.rect)
        text = self.font.render(self.selected, True, WHITE)
        win.blit(text, (self.rect.x + 10, self.rect.y + 10))
        
        
        if self.dropdown_open:
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(self.rect.x, self.rect.y + 40 * (i + 1), self.rect.width, 40)
                pygame.draw.rect(win, BLUE, option_rect)
                option_text = self.font.render(option, True, WHITE)
                win.blit(option_text, (option_rect.x + 10, option_rect.y + 10))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dropdown_open = not self.dropdown_open  

            elif self.dropdown_open:
                for i, option in enumerate(self.options):
                    option_rect = pygame.Rect(self.rect.x, self.rect.y + 40 * (i + 1), self.rect.width, 40)
                    if option_rect.collidepoint(event.pos):
                        self.selected = option
                        self.dropdown_open = False  
                        break 

def main(win, width):
    ROWS = 50
    grid = make_grid(ROWS, width)
    start = None
    end = None
    run = True
    
    maze_options = ["Clear","random", "Maze 1", "Maze 2"]
    maze_dropdown = Dropdown(600, width + 30, 150, maze_options)
    
    options = ["A*", "Dijkstra", "BFS", "DFS"]
    dropdown = Dropdown(20, width + 30, 150, options)
    algorithms = {"A*": algorithmAstart, "Dijkstra": algorithmDijkstra, "BFS": algorithmBFS, "DFS": algorithmDFS}
    
    while run:
        win.fill(WHITE)
        draw(win, grid, ROWS, width)
        
        # Draw each dropdown only once per loop
        dropdown.draw(win)
        maze_dropdown.draw(win)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            dropdown.handle_event(event)
            maze_dropdown.handle_event(event)

            if pygame.mouse.get_pressed()[0]:  # LEFT mouse button
                pos = pygame.mouse.get_pos()
                if is_button_clicked(pos, width // 2 - 120, width + 20, 100, 50):
                    start, end = None, None
                    grid = make_grid(ROWS, width)
                elif is_button_clicked(pos, width // 2 + 20, width + 20, 100, 50) and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    selected_algorithm = algorithms[dropdown.selected]
                    selected_algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)

                try:
                    row, col = get_clicked_pos(pos, ROWS, width)
                    spot = grid[row][col]
                    if not start and spot != end:
                        start = spot
                        start.make_start()
                    elif not end and spot != start:
                        end = spot
                        end.make_end()
                    elif spot != end and spot != start:
                        spot.make_barrier()
                except IndexError:
                    pass

            elif pygame.mouse.get_pressed()[2]:  # RIGHT mouse button
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    selected_algorithm = algorithms[dropdown.selected]
                    selected_algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)

                if event.key == pygame.K_c:
                    start, end = None, None
                    grid = make_grid(ROWS, width)
            if maze_dropdown.selected in MAZE_PRESETS:
                load_maze(grid, maze_dropdown.selected)
                draw(win, grid, ROWS, width)    
        pygame.display.update()  # Update the display only once per loop

    pygame.quit()

main(WIN, WIDTH)

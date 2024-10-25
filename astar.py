import pygame
import random
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
    # Spiral pattern in top left
    (5, 5), (6, 5), (7, 5), (8, 5), (8, 6), (8, 7), (8, 8),
    (7, 8), (6, 8), (5, 8), (5, 7), (5, 6),
    
    # Diagonal barriers
    (10, 2), (11, 3), (12, 4), (13, 5), (14, 6), (15, 7),
    (16, 8), (17, 9), (18, 10), (19, 11), (20, 12),
    
    (30, 2), (31, 3), (32, 4), (33, 5), (34, 6), (35, 7),
    (36, 8), (37, 9), (38, 10), (39, 11), (40, 12),
    
    # Cross pattern in center
    (23, 20), (24, 20), (25, 20), (26, 20), (27, 20),
    (25, 18), (25, 19), (25, 21), (25, 22),
    
    # Zigzag pattern
    (2, 30), (3, 30), (4, 30), (4, 31), (4, 32), (5, 32),
    (6, 32), (6, 33), (6, 34), (7, 34), (8, 34), (8, 35),
    (8, 36), (9, 36), (10, 36),
    
    # Diamond shape
    (40, 40), (41, 39), (42, 38), (43, 37), (44, 36),
    (43, 35), (42, 34), (41, 33), (40, 32), (39, 33),
    (38, 34), (37, 35), (36, 36), (37, 37), (38, 38),
    (39, 39),
    
    # Random scattered walls
    (15, 15), (15, 16), (16, 15),
    (35, 25), (36, 25), (37, 25),
    (20, 45), (21, 45), (22, 45),
    (45, 15), (45, 16), (45, 17),
    
    # L-shaped barriers
    (10, 20), (10, 21), (10, 22), (11, 22), (12, 22),
    (30, 40), (31, 40), (32, 40), (32, 41), (32, 42),
    
    # Small box patterns
    (5, 45), (5, 46), (6, 45), (6, 46),
    (45, 5), (45, 6), (46, 5), (46, 6),
    
    # Additional diagonal lines
    (2, 2), (3, 3), (4, 4),
    (47, 47), (48, 48),
    (2, 48), (3, 47), (4, 46),
    
    # Random obstacles
    (15, 35), (25, 30), (35, 15),
    (12, 12), (38, 42), (42, 38),
    (8, 25), (25, 8), (40, 25),
    
    # Star-like pattern in center-right
    (30, 30), (31, 29), (32, 28), (31, 31), (32, 32),
    (29, 31), (28, 32), (29, 29), (28, 28)
],
    "Maze 2": [
    (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 15), (1, 16), (1, 17),
    (2, 7), (2, 17), (2, 25), (2, 26), (2, 27), (2, 35), (2, 36),
    (3, 7), (3, 17), (3, 27), (3, 36), (3, 42), (3, 43), (3, 44),
    (4, 7), (4, 17), (4, 27), (4, 36), (4, 44),
    (5, 7), (5, 17), (5, 27), (5, 36), (5, 44),
    (6, 7), (6, 17), (6, 27), (6, 36), (6, 44),
    (7, 7), (7, 8), (7, 9), (7, 17), (7, 27), (7, 36), (7, 44),
    (8, 9), (8, 17), (8, 27), (8, 36), (8, 44),
    (9, 9), (9, 17), (9, 27), (9, 36), (9, 44),
    (10, 9), (10, 17), (10, 27), (10, 36), (10, 44),
    (11, 9), (11, 17), (11, 27), (11, 36), (11, 44),
    (12, 9), (12, 17), (12, 27), (12, 36), (12, 44),
    (13, 9), (13, 17), (13, 27), (13, 36), (13, 44),
    (14, 9), (14, 17), (14, 27), (14, 36), (14, 44),
    (15, 9), (15, 17), (15, 27), (15, 36), (15, 44),
    (16, 9), (16, 17), (16, 27), (16, 36), (16, 44),
    (17, 9), (17, 17), (17, 27), (17, 36), (17, 44),
    (18, 9), (18, 17), (18, 27), (18, 36), (18, 44),
    (19, 9), (19, 17), (19, 27), (19, 36), (19, 44),
    (20, 9), (20, 17), (20, 27), (20, 36), (20, 44),
    (21, 9), (21, 17), (21, 27), (21, 36), (21, 44),
    (22, 9), (22, 17), (22, 27), (22, 36), (22, 44),
    (23, 9), (23, 17), (23, 27), (23, 36), (23, 44),
    (24, 9), (24, 17), (24, 27), (24, 36), (24, 44),
    (25, 9), (25, 10), (25, 11), (25, 12), (25, 17), (25, 27), (25, 36), (25, 44),
    (26, 12), (26, 17), (26, 27), (26, 36), (26, 44),
    (27, 12), (27, 17), (27, 27), (27, 36), (27, 44),
    (28, 12), (28, 17), (28, 27), (28, 36), (28, 44),
    (29, 12), (29, 17), (29, 27), (29, 36), (29, 44),
    (30, 12), (30, 17), (30, 27), (30, 36), (30, 44),
    (31, 12), (31, 17), (31, 27), (31, 36), (31, 44),
    (32, 12), (32, 17), (32, 27), (32, 36), (32, 44),
    (33, 12), (33, 17), (33, 27), (33, 36), (33, 44),
    (34, 12), (34, 17), (34, 27), (34, 36), (34, 44),
    (35, 12), (35, 13), (35, 14), (35, 15), (35, 16), (35, 17), (35, 27), (35, 36), (35, 44),
    (36, 17), (36, 27), (36, 36), (36, 44),
    (37, 17), (37, 27), (37, 36), (37, 44),
    (38, 17), (38, 27), (38, 36), (38, 44),
    (39, 17), (39, 27), (39, 36), (39, 44),
    (40, 17), (40, 27), (40, 36), (40, 44),
    (41, 17), (41, 27), (41, 36), (41, 44),
    (42, 17), (42, 27), (42, 36), (42, 44),
    (43, 17), (43, 27), (43, 36), (43, 44),
    (44, 17), (44, 18), (44, 19), (44, 20), (44, 27), (44, 36), (44, 44),
    (45, 20), (45, 27), (45, 36), (45, 44),
    (46, 20), (46, 27), (46, 36), (46, 44),
    (47, 20), (47, 27), (47, 36), (47, 44),
    (48, 20), (48, 27), (48, 36), (48, 44),
    (49, 20), (49, 21), (49, 22), (49, 23), (49, 24), (49, 25), (49, 26), (49, 27)
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
    print("generate_maze")
    # Initialize maze with all walls (1s)
    maze = [[1] * height for _ in range(width)]
    
    def is_near_end(x, y):
        end_x, end_y = width-1, height-1
        # Define how many cells around the end to keep clear (e.g., 2 cells)
        clear_distance = 2
        return (abs(x - end_x) <= clear_distance and 
                abs(y - end_y) <= clear_distance)
        
    def backtrack(x, y):
        maze[x][y] = 0  # Mark current cell as path (0)
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]  
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and maze[nx][ny] == 1:
                maze[x + dx//2][y + dy//2] = 0
                backtrack(nx, ny)
    
    
    backtrack(0, 0)
    
    maze[0][0] = 0
    maze[width-1][height-1] = 0
    
    return maze

def maze_to_coordinates(maze):
    print("maze_to_coordinates")
    coordinates = []
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 1:  # 1 represents barriers/walls
                coordinates.append((i, j))
    return coordinates

def load_maze(grid, maze_name):
    # Clear existing barriers before loading new maze
    for row in grid:
        for spot in row:
            if not spot.is_start() and not spot.is_end():
                spot.reset()
    
    if maze_name == "random":
        # Generate a random maze using the generate_maze function
        maze = generate_maze(50, 50)
        maze_coordinates = maze_to_coordinates(maze)
        print(f"Number of barriers: {len(maze_coordinates)}")  
      
        for (row, col) in maze_coordinates:
            if not (row == 0 and col == 0) and not (row == 49 and col == 49):  # Protect start/end
                grid[row][col].make_barrier()
    
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
   
    options = ["A*", "Dijkstra", "BFS", "DFS"]#maybe add swarm algorithm
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
            
            # Handle maze dropdown events
            previous_selection = maze_dropdown.selected
            maze_dropdown.handle_event(event)
            # Check if selection changed
            if maze_dropdown.selected != previous_selection:
                if maze_dropdown.selected == "random":
                    load_maze(grid, "random")
                    draw(win, grid, ROWS, width)
                elif maze_dropdown.selected in MAZE_PRESETS:
                    load_maze(grid, maze_dropdown.selected)
                    draw(win, grid, ROWS, width)
                elif maze_dropdown.selected == "Clear":
                    start, end = None, None
                    grid = make_grid(ROWS, width)
            
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
                try:
                    row, col = get_clicked_pos(pos, ROWS, width)
                    spot = grid[row][col]
                    spot.reset()
                    if spot == start:
                        start = None
                    elif spot == end:
                        end = None
                except IndexError:
                    pass
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
                    
        pygame.display.update()  
        
    pygame.quit()
    
main(WIN, WIDTH)

import numpy as np
from collections import deque
from PIL import Image
import random
from typing import Tuple, List
from tqdm import tqdm
import os


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def generate_maze(size: int) -> np.ndarray:
    """
    Generates a maze using recursive backtracking.

    Args:
        size: logical grid size (must be odd, e.g., 31)
              The maze will have size x size cells in the final array

    Returns:
        maze: array (size, size) where 0=path, 1=wall
    """
    # Ensure odd size (walls on edges + alternating cells)
    if size % 2 == 0:
        size += 1

    # Initialize everything as wall
    maze = np.ones((size, size), dtype=np.uint8)

    # Starting point (always odd to be a "cell", not a wall)
    start_x, start_y = 1, 1
    maze[start_y, start_x] = 0  # Mark as path

    # Stack for backtracking (iterative DFS to avoid recursion limit)
    stack = [(start_x, start_y)]

    # Directions: (dx, dy) - we move 2 cells at a time (skip the wall)
    directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]  # up, down, left, right

    while stack:
        x, y = stack[-1]

        # Find unvisited neighbors
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Check if within bounds and not visited
            if 0 < nx < size - 1 and 0 < ny < size - 1 and maze[ny, nx] == 1:
                neighbors.append((nx, ny, dx, dy))

        if neighbors:
            # Choose random neighbor
            nx, ny, dx, dy = random.choice(neighbors)

            # Remove wall between current cell and neighbor
            maze[y + dy // 2, x + dx // 2] = 0

            # Mark neighbor as visited
            maze[ny, nx] = 0

            # Add neighbor to stack
            stack.append((nx, ny))
        else:
            # No unvisited neighbors, backtrack
            stack.pop()

    return maze


def solve_maze(maze: np.ndarray) -> List[Tuple[int, int]]:
    """
    Finds the shortest path from the top-left corner
    to the bottom-right corner using BFS.

    Args:
        maze: array (size, size) where 0=path, 1=wall

    Returns:
        path: list of (x, y) coordinates of the solution path
              empty list if no solution exists
    """
    size = maze.shape[0]

    # Entry: first accessible cell at top-left
    # Exit: last accessible cell at bottom-right
    start = (1, 1)
    end = (size - 2, size - 2)

    # BFS
    queue = deque([(start, [start])])  # (current_position, path_so_far)
    visited = {start}

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right

    while queue:
        (x, y), path = queue.popleft()

        if (x, y) == end:
            return path

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if (0 <= nx < size and 0 <= ny < size and
                maze[ny, nx] == 0 and (nx, ny) not in visited):

                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))

    return []  # No solution (shouldn't happen with valid mazes)


def render_maze(maze: np.ndarray,
                path: List[Tuple[int, int]] = None,
                output_size: int = 64) -> np.ndarray:
    """
    Renders the maze as an RGB image.

    Args:
        maze: array (size, size) where 0=path, 1=wall
        path: list of solution coordinates (optional)
        output_size: final image size (default 64)

    Returns:
        image: array (output_size, output_size, 3) RGB uint8
    """
    size = maze.shape[0]

    # Create RGB image
    # Wall = black, Path = white
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[maze == 0] = [255, 255, 255]  # Free path = white
    image[maze == 1] = [0, 0, 0]        # Wall = black

    # Mark entry and exit
    image[1, 1] = [0, 255, 0]                    # Entry = green
    image[size - 2, size - 2] = [0, 255, 0]     # Exit = green

    # Mark solution if provided
    if path:
        for x, y in path:
            if (x, y) != (1, 1) and (x, y) != (size - 2, size - 2):
                image[y, x] = [255, 0, 0]  # Solution = red

    # Resize to output_size x output_size
    img_pil = Image.fromarray(image)
    img_pil = img_pil.resize((output_size, output_size), Image.NEAREST)

    return np.array(img_pil)


def generate_maze_pair(size: int = None, output_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates an (input, output) maze pair.

    Args:
        size: logical maze size (None = random among options)
        output_size: final image size

    Returns:
        input_img: maze without solution (H, W, 3)
        output_img: maze with solution marked (H, W, 3)
    """
    # Vary size for diversity
    if size is None:
        size = random.choice([11, 15, 21, 25, 31])

    # Generate maze
    maze = generate_maze(size)

    # Solve
    path = solve_maze(maze)

    # Render
    input_img = render_maze(maze, path=None, output_size=output_size)
    output_img = render_maze(maze, path=path, output_size=output_size)

    return input_img, output_img


def generate_batch(batch_size: int, output_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a batch of maze pairs.

    Returns:
        inputs: (batch_size, output_size, output_size, 3)
        outputs: (batch_size, output_size, output_size, 3)
    """
    inputs = []
    outputs = []

    for _ in tqdm(range(batch_size), desc="Generating mazes"):
        inp, out = generate_maze_pair(output_size=output_size)
        inputs.append(inp)
        outputs.append(out)

    return np.stack(inputs), np.stack(outputs)


def generate_dataset(total_size: int,
                     batch_size: int = 10000,
                     output_size: int = 64,
                     save_dir: str = './data'):
    """
    Generates the complete dataset and saves in batches.

    Args:
        total_size: total number of pairs (e.g., 1_000_000)
        batch_size: size of each .npz file
        output_size: image resolution
        save_dir: directory to save to
    """
    os.makedirs(save_dir, exist_ok=True)

    num_batches = total_size // batch_size

    print(f"Generating {total_size:,} pairs in {num_batches} batches of {batch_size:,}")
    print(f"Saving to: {save_dir}")
    print("-" * 50)

    for batch_idx in range(num_batches):
        print(f"\nBatch {batch_idx + 1}/{num_batches}")

        # Generate the batch
        inputs, outputs = generate_batch(batch_size, output_size)

        # Save as compressed .npz
        filename = os.path.join(save_dir, f'batch_{batch_idx:04d}.npz')
        np.savez_compressed(filename, inputs=inputs, outputs=outputs)

        # Check file size
        file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
        print(f"Saved: {filename} ({file_size:.1f} MB)")

        # Free memory
        del inputs, outputs

    print("\n" + "=" * 50)
    print("Dataset complete!")
    print(f"Total: {total_size:,} pairs")
    print(f"Files: {num_batches} batches in {save_dir}")

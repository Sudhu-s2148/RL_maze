# 🧩 RL Maze Visualizer

An interactive reinforcement learning maze visualizer built with Python and Pygame. Watch trained agents navigate through various procedurally generated maze configurations.

---

## 🚀 Prerequisites: Install `uv`

This project uses **`uv`** for lightning-fast dependency management and automatic Python version handling. You don't even need Python pre-installed on your machine—`uv` will handle it for you.

Run the installer for your operating system:

* **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
* **macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

*(After installing, restart your terminal to ensure the `uv` command is recognized.)*

---

## 📥 Getting the Project

Choose **one** of the two methods below to get the project onto your local machine:

### Option A: Clone via Git (Recommended)
If you have Git installed, open your terminal and run:
```bash
git clone https://github.com/Sudhu-s2148/RL_maze.git
cd RL_maze
```

### Option B: Download as a ZIP
1. Go to the GitHub repository page.
2. Click the green **Code** button and select **Download ZIP**.
3. Extract the ZIP file on your computer.
4. Open your terminal and navigate inside the extracted folder:
   ```bash
   cd path/to/RL_maze-main
   ```

---

## 🎮 Running the Visualizer

`uv` automatically handles creating the virtual environment and installing dependencies on your first run. Use the `--maze` command-line argument to pick your maze configuration:

### Available Maze Setups

* **Small 5x4 Grid:**
  ```bash
  uv run agent_visuals3.0.py --maze 5x4
  ```

* **20x20 Backtracker Maze:**
  ```bash
  uv run agent_visuals3.0.py --maze 20x20-backtracker
  ```

* **20x20 Rooms Maze:**
  ```bash
  uv run agent_visuals3.0.py --maze 20x20-rooms
  ```

* **Large 50x50 Rooms Maze:**
  ```bash
  uv run agent_visuals3.0.py --maze 50x50-rooms
  ```

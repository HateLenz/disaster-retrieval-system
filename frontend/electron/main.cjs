const { app, BrowserWindow } = require("electron");
const { spawn } = require("child_process");
const path = require("path");

let backendProcess = null;

function startBackend() {
  if (process.env.DFR_ELECTRON_START_BACKEND === "0") {
    return;
  }

  const repoRoot = path.resolve(__dirname, "..", "..");
  const command = process.env.DFR_BACKEND_CMD || "python";
  const args = process.env.DFR_BACKEND_ARGS
    ? process.env.DFR_BACKEND_ARGS.split(" ")
    : ["-m", "uvicorn", "src.api.app:app", "--host", "127.0.0.1", "--port", "8000"];

  backendProcess = spawn(command, args, {
    cwd: repoRoot,
    env: {
      ...process.env,
      DFR_LOCAL_FILES_ONLY: process.env.DFR_LOCAL_FILES_ONLY || "1"
    },
    windowsHide: true,
    stdio: "ignore"
  });

  backendProcess.on("exit", () => {
    backendProcess = null;
  });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1360,
    height: 860,
    minWidth: 1080,
    minHeight: 720,
    backgroundColor: "#f5f7f5",
    title: "灾后检索系统",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  const devUrl = process.env.VITE_DEV_SERVER_URL || "http://127.0.0.1:5173";
  if (process.env.NODE_ENV === "development" || process.defaultApp) {
    win.loadURL(devUrl);
  } else {
    win.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  }
}

app.whenReady().then(() => {
  startBackend();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  if (backendProcess) {
    backendProcess.kill();
  }
});

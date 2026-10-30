import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// 开发时 Vite 跑在 5173，接口在 8000。**用代理而不是给 FastAPI 加 CORS**：
// 代理让浏览器看到的始终是同一个 origin，生产（FastAPI 直接托管 /ui）和开发
// 的行为一致；开 CORS 则是为了开发环境改生产接口的信任边界，方向反了。
const target = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  // 生产由 FastAPI 挂在 /ui 下，静态资源的绝对路径必须带这个前缀，
  // 否则 /ui/assets/index-xxx.js 会 404——而页面本身还能打开，看起来只是白屏。
  base: "/ui/",
  server: {
    port: 5173,
    proxy: {
      "/health": target,
      "/chat": target,
      "/trace": target,
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});

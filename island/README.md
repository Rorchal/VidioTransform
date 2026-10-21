# 海岛动画（Blender 程序化生成）

一个完全由代码生成的清新风格热带海岛航拍镜头：碧绿泻湖、白沙滩、椰子树、灌木、礁石和云朵，
相机环绕上升。没有任何外部素材，所有几何体、材质和动画都在 `build_island.py` 里生成。

## 依赖

```bash
pip install bpy numpy          # bpy = 无界面的 Blender（5.x）
apt-get install -y libegl1 libgl1 libgles2 libglu1-mesa libgl1-mesa-dri ffmpeg   # 无 GPU 容器需要的软件 OpenGL
```

## 使用

```bash
# 单帧预览（640x360）
LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless python3 -I build_island.py --quality preview --frame 110 --out renders/test

# 完整动画（默认 1280x720，192 帧 @ 24fps = 8 秒），每帧 CPU 时间上限 40 秒
./render.sh medium renders/medium

# 合成 mp4
./make_video.sh renders/medium island.mp4 24
```

`--quality final` 输出 1920x1080。`--save-blend island.blend` 可以把场景保存出来在 Blender 界面里继续调整。

## 场景结构

- 地形：numpy 分形噪声高度场 + 域扭曲，海平面附近压平成沙滩，外围有一圈浅水礁盘再陡降到深海。
- 水面：Ocean 修改器驱动的波浪 + 透射材质 + 体积吸收（浅处碧绿、深处湛蓝），阴影光线直接穿透水面模拟焦散。
- 植被：贝塞尔曲线椰子树干 + 程序化下垂叶片，灌木为成簇的球体。
- 天空：基于视线方向的渐变（地平线浅青 → 天顶深蓝），太阳灯提供主光；云朵不投影、不参与倒影。
- 相机：三段贝塞尔路径环绕岛屿并抬升，Track To 约束始终对准岛心。

"use strict";

// Display-only: ray-cast the recorded OCT tensor, not a point surrogate.
// Every mode uses one scanner mapping, camera, zoom and viewport.
(() => {
  const canvas = document.querySelector("#representation-canvas");
  const status = document.querySelector("#representation-loading");
  const gl = canvas.getContext("webgl2", {
    alpha: false,
    antialias: true,
    preserveDrawingBuffer: true,
  });
  if (!gl) {
    status.textContent = "Interactive volume rendering requires WebGL 2.";
    return;
  }
  function program(vertex, fragment) {
    const p = gl.createProgram();
    for (const [kind, source] of [
      [gl.VERTEX_SHADER, vertex],
      [gl.FRAGMENT_SHADER, fragment],
    ]) {
      const shader = gl.createShader(kind);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
        throw new Error(gl.getShaderInfoLog(shader));
      gl.attachShader(p, shader);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(p));
    return p;
  }
  const volumeProgram = program(
    `#version 300 es
    in vec2 position; out vec2 screen;
    void main(){screen=position;gl_Position=vec4(position,0.,1.);}`,
    `#version 300 es
    precision highp float; precision highp sampler3D; precision highp sampler2D;
    in vec2 screen; out vec4 color;
    uniform sampler3D intensity; uniform sampler3D support;
    uniform sampler2D transfer;
    uniform vec3 normal; uniform vec3 right; uniform vec3 up;
    uniform float scale; uniform float aspect; uniform bool tissueOnly;
    uniform float spacing; uniform float focalZ; uniform float stepSize;
    uniform float referenceStep; uniform float airGain; uniform float airBrightness;
    uniform vec3 background;
    void main(){
      vec3 plane=scale*(screen.x*aspect*right+screen.y*up);
      float zmax=255.*spacing;
      float lateral=5.*(abs(normal.x)+abs(normal.y));
      float nearDepth=lateral+max(normal.z*focalZ,normal.z*(focalZ-zmax));
      float farDepth=-lateral+min(normal.z*focalZ,normal.z*(focalZ-zmax));
      vec4 accumulated=vec4(0.);
      for(int i=0;i<384;i++){
        float depth=nearDepth+stepSize-float(i)*stepSize;
        if(depth<farDepth-stepSize) break;
        vec3 p=plane+normal*depth;
        vec3 world=vec3(p.xy+5.,focalZ-p.z);
        if(any(lessThan(world,vec3(0.)))||any(greaterThan(world,vec3(10.,10.,zmax)))) continue;
        vec3 uv=(world/spacing+.5)/vec3(256.,256.,256.);
        float value=texture(intensity,uv).r;
        float mask=texture(support,uv).r;
        if(tissueOnly){
          value/=max(mask,1e-6);
        }
        int index=int(clamp((value+6.)/11.*4095.,0.,4095.));
        vec4 lut=texelFetch(transfer,ivec2(index,0),0);
        float air=texelFetch(transfer,ivec2(index,1),0).r;
        // Keep measured air throughout the cube while suppressing bright
        // non-tissue filaments/plate returns in the background opacity only.
        air*=1.-smoothstep(.3,1.,value);
        float referenceAlpha=tissueOnly?lut.a*mask:lut.a*mask+air*airGain*(1.-mask);
        float opacity=1.-pow(1.-clamp(referenceAlpha,0.,.95),stepSize/referenceStep);
        vec3 rgb=tissueOnly?lut.rgb:clamp(lut.rgb*(mask+(1.-mask)*airBrightness),0.,1.);
        accumulated.rgb+=(1.-accumulated.a)*opacity*rgb;
        accumulated.a+=(1.-accumulated.a)*opacity;
        if(accumulated.a>.99999) break;
      }
      color=vec4(accumulated.rgb+(1.-accumulated.a)*background,1.);
    }`,
  );
  const pointProgram = program(
    `#version 300 es
    in vec3 position; uniform vec3 right; uniform vec3 up; uniform vec3 normal;
    uniform float scale; uniform float aspect; uniform float pointSize; uniform float focalZ;
    void main(){
      // XYZ normalized on the full scanner grid. All modes share its center;
      // never independently fit point bounds.
      vec3 p=vec3(position.xy*5.,focalZ-(position.z+1.)*5.);
      gl_Position=vec4(dot(p,right)/(scale*aspect),dot(p,up)/scale,-dot(p,normal)/24.,1.);
      gl_PointSize=pointSize;
    }`,
    `#version 300 es
    precision highp float; out vec4 color;
    void main(){float r=length(gl_PointCoord-.5);if(r>.5)discard;
      color=vec4(.47843,.89412,.91373,1.-smoothstep(.32,.5,r));}`,
  );
  const boxProgram = program(
    `#version 300 es
    in vec3 position; uniform vec3 right;uniform vec3 up;uniform vec3 normal;
    uniform float scale;uniform float aspect;uniform float focalZ;
    void main(){vec3 p=vec3(position.xy-5.,focalZ-position.z);
      gl_Position=vec4(dot(p,right)/(scale*aspect),dot(p,up)/scale,-dot(p,normal)/24.,1.);}`,
    `#version 300 es
    precision highp float;out vec4 color;void main(){color=vec4(.25882,.27059,.36078,1.);}`,
  );
  const quad = gl.createVertexArray();
  gl.bindVertexArray(quad);
  const quadBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
    gl.STATIC_DRAW,
  );
  const quadLocation = gl.getAttribLocation(volumeProgram, "position");
  gl.enableVertexAttribArray(quadLocation);
  gl.vertexAttribPointer(quadLocation, 2, gl.FLOAT, false, 0, 0);
  const box = gl.createVertexArray();
  gl.bindVertexArray(box);
  const corners = [];
  for (const x of [0, 10])
    for (const y of [0, 10])
      for (const z of [0, 10]) corners.push([x, y, z]);
  const edges = [];
  for (let i = 0; i < 8; i++)
    for (let j = i + 1; j < 8; j++)
      if (corners[i].filter((v, k) => v !== corners[j][k]).length === 1)
        edges.push(...corners[i], ...corners[j]);
  const edgeBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, edgeBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(edges), gl.STATIC_DRAW);
  const boxLocation = gl.getAttribLocation(boxProgram, "position");
  gl.enableVertexAttribArray(boxLocation);
  gl.vertexAttribPointer(boxLocation, 3, gl.FLOAT, false, 0, 0);
  let active = "full",
    azimuth = (15 * Math.PI) / 180,
    elevation = (18 * Math.PI) / 180,
    scale = 6.8;
  let style = {
    display_background_rgb: [0.025, 0.029, 0.051],
    focal_point: [5, 5, 5],
    spacing: 10 / 255,
    ray_step: 0.06,
    opacity_reference_step: 0.078,
    air_gain: 1.8,
    air_brightness: 2.2,
  };
  let volumeTexture,
    tissueTexture,
    maskTexture,
    transferTexture,
    points,
    count = 0,
    volumePending,
    pointPending,
    queued = false;
  function texture(data, unit, filter, half = false) {
    const t = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_3D, t);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, filter);
    for (const axis of [
      gl.TEXTURE_WRAP_S,
      gl.TEXTURE_WRAP_T,
      gl.TEXTURE_WRAP_R,
    ])
      gl.texParameteri(gl.TEXTURE_3D, axis, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(
      gl.TEXTURE_3D,
      0,
      half ? gl.R16F : gl.R8,
      256,
      256,
      256,
      0,
      gl.RED,
      half ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE,
      data,
    );
    return t;
  }
  async function binary(name, length) {
    const response = await fetch("assets/" + name + "?v=20261006-clear");
    if (!response.ok) throw new Error("OCT display data unavailable");
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.length !== length)
      throw new Error("Unexpected OCT display shape");
    return bytes;
  }
  function loadVolume() {
    const size = 256 * 256 * 256;
    return (volumePending ||= Promise.all([
      binary("paper-display-volume-f16.bin", size * 2),
      binary("paper-display-tissue-f16.bin", size * 2),
      binary("paper-display-mask.bin", size),
      binary("paper-transfer-lut.bin", 4096 * 2 * 4 * 4),
      fetch("assets/paper-render-style.json?v=20261006-clear").then((r) => r.json()),
    ]).then(([volume, tissue, mask, lut, settings]) => {
      style = settings;
      volumeTexture = texture(
        new Uint16Array(volume.buffer),
        0,
        gl.LINEAR,
        true,
      );
      tissueTexture = texture(
        new Uint16Array(tissue.buffer),
        0,
        gl.LINEAR,
        true,
      );
      for (let i = 0; i < mask.length; i++) mask[i] *= 255;
      maskTexture = texture(mask, 1, gl.LINEAR);
      transferTexture = gl.createTexture();
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, transferTexture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA32F,
        4096,
        2,
        0,
        gl.RGBA,
        gl.FLOAT,
        new Float32Array(lut.buffer),
      );
    }));
  }
  function loadPoints() {
    return (pointPending ||= fetch("assets/point-cloud.json")
      .then((r) => {
        if (!r.ok) throw new Error("Points unavailable");
        return r.json();
      })
      .then((data) => {
        if (data.positions.length !== 1024 * 3)
          throw new Error("Unexpected point count");
        points = gl.createVertexArray();
        gl.bindVertexArray(points);
        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(
          gl.ARRAY_BUFFER,
          new Float32Array(data.positions),
          gl.STATIC_DRAW,
        );
        const location = gl.getAttribLocation(pointProgram, "position");
        gl.enableVertexAttribArray(location);
        gl.vertexAttribPointer(location, 3, gl.FLOAT, false, 0, 0);
        count = data.positions.length / 3;
      }));
  }
  function camera(p, aspect) {
    const normal = [
      Math.cos(elevation) * Math.cos(azimuth),
      Math.cos(elevation) * Math.sin(azimuth),
      Math.sin(elevation),
    ];
    const right = [Math.sin(azimuth), -Math.cos(azimuth), 0];
    const up = [
      -Math.sin(elevation) * Math.cos(azimuth),
      -Math.sin(elevation) * Math.sin(azimuth),
      Math.cos(elevation),
    ];
    for (const [name, value] of [
      ["normal", normal],
      ["right", right],
      ["up", up],
    ])
      gl.uniform3fv(gl.getUniformLocation(p, name), value);
    gl.uniform1f(gl.getUniformLocation(p, "scale"), scale);
    gl.uniform1f(gl.getUniformLocation(p, "aspect"), aspect);
    gl.uniform1f(gl.getUniformLocation(p, "focalZ"), style.focal_point[2]);
  }
  function draw() {
    queued = false;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const ratio = Math.min(devicePixelRatio, 1.5);
    canvas.width = Math.round(rect.width * ratio);
    canvas.height = Math.round(rect.height * ratio);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(...style.display_background_rgb, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (active === "points") {
      if (!points) return;
      gl.useProgram(pointProgram);
      camera(pointProgram, rect.width / rect.height);
      gl.uniform1f(
        gl.getUniformLocation(pointProgram, "pointSize"),
        4.3 * ratio,
      );
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindVertexArray(points);
      gl.drawArrays(gl.POINTS, 0, count);
    } else {
      if (!volumeTexture || !maskTexture) return;
      gl.useProgram(volumeProgram);
      camera(volumeProgram, rect.width / rect.height);
      gl.disable(gl.BLEND);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(
        gl.TEXTURE_3D,
        active === "tissue" ? tissueTexture : volumeTexture,
      );
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_3D, maskTexture);
      gl.uniform1i(gl.getUniformLocation(volumeProgram, "intensity"), 0);
      gl.uniform1i(gl.getUniformLocation(volumeProgram, "support"), 1);
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, transferTexture);
      gl.uniform1i(gl.getUniformLocation(volumeProgram, "transfer"), 2);
      for (const [name, value] of [
        ["spacing", style.spacing],
        ["stepSize", style.ray_step],
        ["referenceStep", style.opacity_reference_step],
        ["airGain", style.air_gain],
        ["airBrightness", style.air_brightness],
      ])
        gl.uniform1f(gl.getUniformLocation(volumeProgram, name), value);
      gl.uniform3fv(
        gl.getUniformLocation(volumeProgram, "background"),
        style.display_background_rgb,
      );
      gl.uniform1i(
        gl.getUniformLocation(volumeProgram, "tissueOnly"),
        active === "tissue",
      );
      gl.bindVertexArray(quad);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      if (active === "full") {
        gl.useProgram(boxProgram);
        camera(boxProgram, rect.width / rect.height);
        gl.bindVertexArray(box);
        gl.drawArrays(gl.LINES, 0, edges.length / 3);
      }
    }
  }
  function invalidate() {
    if (!queued) {
      queued = true;
      requestAnimationFrame(draw);
    }
  }
  async function select(key) {
    active = key;
    status.hidden = false;
    status.textContent = "Loading recorded OCT…";
    try {
      await (key === "points" ? loadPoints() : loadVolume());
      if (active === key) {
        status.hidden = true;
        invalidate();
      }
    } catch (error) {
      status.textContent = "3D data could not load. Please reload the page.";
      console.error(error);
    }
  }
  window.octViewer = { setRepresentation: select };
  let drag = null;
  canvas.addEventListener("pointerdown", (event) => {
    canvas.focus();
    drag = [event.clientX, event.clientY];
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag) return;
    azimuth -= (event.clientX - drag[0]) * 0.008;
    elevation = Math.max(
      -1.5,
      Math.min(1.5, elevation + (event.clientY - drag[1]) * 0.008),
    );
    drag = [event.clientX, event.clientY];
    invalidate();
  });
  for (const event of ["pointerup", "pointercancel"])
    canvas.addEventListener(event, () => {
      drag = null;
    });
  canvas.addEventListener(
    "wheel",
    (event) => {
      if (document.activeElement !== canvas) return;
      event.preventDefault();
      scale = Math.max(
        2.2,
        Math.min(12, scale * Math.exp(event.deltaY * 0.001)),
      );
      invalidate();
    },
    { passive: false },
  );
  canvas.addEventListener("keydown", (event) => {
    if (
      !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "-"].includes(
        event.key,
      )
    )
      return;
    event.preventDefault();
    if (event.key === "ArrowLeft") azimuth -= 0.1;
    if (event.key === "ArrowRight") azimuth += 0.1;
    if (event.key === "ArrowUp") elevation = Math.min(1.5, elevation + 0.1);
    if (event.key === "ArrowDown") elevation = Math.max(-1.5, elevation - 0.1);
    if (event.key === "+") scale = Math.max(2.2, scale * 0.92);
    if (event.key === "-") scale = Math.min(12, scale / 0.92);
    invalidate();
  });
  document.querySelector("#reset-view").addEventListener("click", () => {
    azimuth = (15 * Math.PI) / 180;
    elevation = (18 * Math.PI) / 180;
    scale = 6.8;
    invalidate();
  });
  new ResizeObserver(invalidate).observe(canvas);
  // Do not make the larger interactive arrays compete with the opening video.
  const visible = new IntersectionObserver(
    (entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        visible.disconnect();
        select(active);
      }
    },
    { rootMargin: "160px" },
  );
  visible.observe(canvas);
})();

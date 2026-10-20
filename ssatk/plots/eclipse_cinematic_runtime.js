// V22.2 cinematic eclipse runtime — exact opaque bodies, full-disc weighted photospheric paths, physical-UTC playback, body-following north-up cameras, capability-negotiated postprocessing, and explicit provenance. THREE, CONFIG, EARTH_TEXTURE_URI, MOON_TEXTURE_URI, and PHOTOMETRY_LUT_B64 are injected by cinematic_light_3d.py.

const UNIT_KM = CONFIG.unit_km;
const root = document.getElementById('viewport');
const canvas = document.getElementById('gl');
const statusEl = document.getElementById('status');
const phaseEl = document.getElementById('phase');
const timeEl = document.getElementById('timeLabel');
const slider = document.getElementById('timeline');
const playButton = document.getElementById('play');
const rayToggle = document.getElementById('raysToggle');
const boundaryToggle = document.getElementById('boundaryToggle');
const atmosphereToggle = document.getElementById('atmosphereToggle');
const starToggle = document.getElementById('starToggle');
const labelsToggle = document.getElementById('labelsToggle');
const scientificToggle = document.getElementById('scientificToggle');
const orbitToggle = document.getElementById('orbitToggle');
const photometrySelect = document.getElementById('photometrySelect');
const FIXED_BLOOM = 1.0;
const FIXED_EXPOSURE = 1.05;

let renderer;
let rendererFallback = false;
function createRenderer(options) { return new THREE.WebGLRenderer(options); }
try {
  renderer = createRenderer({
    canvas, antialias: true, alpha: false, powerPreference: 'high-performance',
    logarithmicDepthBuffer: true, preserveDrawingBuffer: true,
  });
} catch (primaryError) {
  try {
    rendererFallback = true;
    renderer = createRenderer({
      canvas, antialias: false, alpha: false, powerPreference: 'default',
      logarithmicDepthBuffer: true, preserveDrawingBuffer: true,
    });
  } catch (fallbackError) {
    statusEl.textContent = 'WebGL2 could not be initialized and the WebGL1 fallback also failed. Open this file in a current hardware-accelerated browser.';
    statusEl.classList.add('error');
    throw fallbackError;
  }
}
const gl = renderer.getContext();
const isWebGL2 = renderer.capabilities.isWebGL2;
const maxSamples = isWebGL2 && gl.getParameter ? Number(gl.getParameter(gl.MAX_SAMPLES) || 0) : 0;
const hasHalfFloatColor = Boolean(isWebGL2 || renderer.extensions.has('EXT_color_buffer_half_float') || renderer.extensions.has('EXT_color_buffer_float'));
const renderTargetType = hasHalfFloatColor ? THREE.HalfFloatType : THREE.UnsignedByteType;
let postprocessingEnabled = true;
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;
renderer.autoClear = false;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x01030a);
const camera = new THREE.PerspectiveCamera(38, 1, 0.002, 800000.0);
camera.up.set(0, 0, 1);

const mainGroup = new THREE.Group();
scene.add(mainGroup);
const bodyGroup = new THREE.Group();
const atmosphereGroup = new THREE.Group();
const localRayGroup = new THREE.Group();
const fullRayGroup = new THREE.Group();
const boundaryGroup = new THREE.Group();
const labelGroup = new THREE.Group();
const scientificGroup = new THREE.Group();
const orbitGroup = new THREE.Group();
mainGroup.add(bodyGroup, atmosphereGroup, localRayGroup, fullRayGroup, boundaryGroup, labelGroup, scientificGroup, orbitGroup);

function setStatus(text, isError=false) {
  statusEl.textContent = text;
  statusEl.classList.toggle('error', isError);
}

function makeTexture(uri, anisotropy=8) {
  const texture = new THREE.TextureLoader().load(uri, () => setStatus('Ready — body north is screen-up in public views; drag to orbit, wheel to zoom'));
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.anisotropy = Math.min(renderer.capabilities.getMaxAnisotropy(), anisotropy);
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  return texture;
}

const earthTexture = makeTexture(EARTH_TEXTURE_URI, 16);
const moonTexture = makeTexture(MOON_TEXTURE_URI, 16);

function decodeBase64Bytes(text) {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; ++i) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
function makePhotometryCorrectionTexture() {
  const meta = CONFIG.photometry_lut;
  if (!meta || !PHOTOMETRY_LUT_B64) throw new Error('Missing V19 photometry correction LUT');
  const raw = decodeBase64Bytes(PHOTOMETRY_LUT_B64);
  const expected = Number(meta.width) * Number(meta.height);
  if (raw.length !== expected) throw new Error(`Photometry LUT byte count ${raw.length} does not match ${expected}`);
  const rgba = new Uint8Array(expected * 4);
  for (let i = 0; i < expected; ++i) {
    rgba[i*4] = raw[i]; rgba[i*4+1] = 0; rgba[i*4+2] = 0; rgba[i*4+3] = 255;
  }
  const texture = new THREE.DataTexture(rgba, Number(meta.width), Number(meta.height), THREE.RGBAFormat, THREE.UnsignedByteType);
  texture.colorSpace = THREE.NoColorSpace;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.flipY = false;
  texture.needsUpdate = true;
  return texture;
}
const photometryCorrectionTexture = makePhotometryCorrectionTexture();
const photometryLutRange = new THREE.Vector4(
  Number(CONFIG.photometry_lut.q_max), Number(CONFIG.photometry_lut.s_max),
  Number(CONFIG.photometry_lut.delta_min), Number(CONFIG.photometry_lut.delta_max)
);
const photometryLutSize = new THREE.Vector2(Number(CONFIG.photometry_lut.width), Number(CONFIG.photometry_lut.height));

function makeLatLonGeometry(latSegments, lonSegments, axes) {
  // One vertex per pole plus seam-closed intermediate rings.  The former
  // all-longitudes-at-each-pole topology generated thousands of zero-area
  // triangles that could fold or sparkle on some GPU/index paths.
  const positions = [], normals = [], uvs = [], indices = [];
  const a = axes[0], b = axes[1], c = axes[2];
  function addVertex(x,y,z,u,v){
    positions.push(x,y,z);
    const nx=x/(a*a),ny=y/(b*b),nz=z/(c*c),inv=1/Math.hypot(nx,ny,nz);
    normals.push(nx*inv,ny*inv,nz*inv);uvs.push(u,v);
  }
  const northIndex=0;
  addVertex(0,0,c,0.5,0.0);
  const row=lonSegments+1;
  for(let i=1;i<latSegments;i++){
    const v=i/latSegments,lat=Math.PI*(0.5-v),cl=Math.cos(lat),sl=Math.sin(lat);
    for(let j=0;j<=lonSegments;j++){
      const u=j/lonSegments,lon=-Math.PI+u*Math.PI*2,co=Math.cos(lon),so=Math.sin(lon);
      addVertex(a*cl*co,b*cl*so,c*sl,u,v);
    }
  }
  const southIndex=positions.length/3;
  addVertex(0,0,-c,0.5,1.0);
  const firstRing=1;
  for(let j=0;j<lonSegments;j++) indices.push(northIndex,firstRing+j,firstRing+j+1);
  for(let i=0;i<latSegments-2;i++){
    const upper=firstRing+i*row,lower=upper+row;
    for(let j=0;j<lonSegments;j++){
      const p0=upper+j,p1=p0+1,p2=lower+j,p3=p2+1;
      indices.push(p0,p2,p1,p1,p2,p3);
    }
  }
  const lastRing=firstRing+(latSegments-2)*row;
  for(let j=0;j<lonSegments;j++) indices.push(southIndex,lastRing+j+1,lastRing+j);
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  geometry.setAttribute('normal',new THREE.Float32BufferAttribute(normals,3));
  geometry.setAttribute('uv',new THREE.Float32BufferAttribute(uvs,2));
  geometry.setIndex(indices);geometry.computeBoundingSphere();
  geometry.userData={northIndex,southIndex,degeneratePoleTriangles:0};
  return geometry;
}

const commonVertexShader = `
  #include <common>
  #include <logdepthbuf_pars_vertex>
  varying vec2 vUv;
  varying vec3 vWorldPosition;
  varying vec3 vWorldNormal;
  void main() {
    vUv = uv;
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vWorldPosition = wp.xyz;
    vWorldNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * viewMatrix * wp;
    #include <logdepthbuf_vertex>
  }
`;

const bodyFragmentShader = `
  precision highp float;
  #include <common>
  #include <logdepthbuf_pars_fragment>
  uniform sampler2D uMap;
  uniform vec3 uSunPosition;
  uniform vec3 uOccluderPosition;
  uniform float uSunRadius;
  uniform float uOccluderRadius;
  uniform float uEclipseEnabled;
  uniform float uIsEarth;
  uniform float uLunarRed;
  uniform float uExposure;
  uniform float uNightFloor;
  uniform float uObserverSilhouette;
  uniform float uPhotometryMode;
  uniform sampler2D uPhotometryLut;
  uniform vec4 uPhotometryLutRange;
  uniform vec2 uPhotometryLutSize;
  varying vec2 vUv;
  varying vec3 vWorldPosition;
  varying vec3 vWorldNormal;

  float safeAcos(float x) { return acos(clamp(x, -1.0, 1.0)); }
  float circleVisibility(float rOcc, float rSun, float d) {
    if (d >= rOcc + rSun) return 1.0;
    if (d <= abs(rOcc - rSun)) {
      if (rOcc >= rSun) return 0.0;
      return clamp(1.0 - (rOcc * rOcc) / max(rSun * rSun, 1e-12), 0.0, 1.0);
    }
    float dSafe = max(d, 1e-9);
    float a1 = clamp((dSafe*dSafe + rOcc*rOcc - rSun*rSun) / (2.0*dSafe*rOcc), -1.0, 1.0);
    float a2 = clamp((dSafe*dSafe + rSun*rSun - rOcc*rOcc) / (2.0*dSafe*rSun), -1.0, 1.0);
    float term = max((-dSafe+rOcc+rSun)*(dSafe+rOcc-rSun)*(dSafe-rOcc+rSun)*(dSafe+rOcc+rSun), 0.0);
    float overlap = rOcc*rOcc*acos(a1) + rSun*rSun*acos(a2) - 0.5*sqrt(term);
    return clamp(1.0 - overlap / max(3.141592653589793*rSun*rSun, 1e-12), 0.0, 1.0);
  }
  float limbDarkenedVisibility(float rOcc, float rSun, float d) {
    float uniformVisibility = circleVisibility(rOcc, rSun, d);
    if (d >= rOcc + rSun) return 1.0;
    if (rOcc >= rSun && d <= rOcc - rSun) return 0.0;
    float q = rOcc / max(rSun, 1.0e-9);
    float ss = d / max(rSun, 1.0e-9);
    if (q < 0.0 || ss < 0.0 || q > uPhotometryLutRange.x || ss > uPhotometryLutRange.y) return uniformVisibility;
    vec2 grid = vec2(ss / uPhotometryLutRange.y, q / uPhotometryLutRange.x);
    vec2 uv = (grid * (uPhotometryLutSize - vec2(1.0)) + vec2(0.5)) / uPhotometryLutSize;
    float encodedCorrection = texture2D(uPhotometryLut, uv).r;
    float correction = mix(uPhotometryLutRange.z, uPhotometryLutRange.w, encodedCorrection);
    return clamp(uniformVisibility + correction, 0.0, 1.0);
  }
  float finiteSunVisibility(vec3 p) {
    if (uEclipseEnabled < 0.5) return 1.0;
    vec3 s = uSunPosition - p;
    vec3 o = uOccluderPosition - p;
    float ds = length(s);
    float doo = length(o);
    float rSun = asin(clamp(uSunRadius / max(ds, uSunRadius), 0.0, 1.0));
    float rOcc = asin(clamp(uOccluderRadius / max(doo, uOccluderRadius), 0.0, 1.0));
    float sep = safeAcos(dot(normalize(s), normalize(o)));
    return uPhotometryMode > 0.5 ? limbDarkenedVisibility(rOcc, rSun, sep) : circleVisibility(rOcc, rSun, sep);
  }
  void main() {
    #include <logdepthbuf_fragment>
    vec3 albedo = texture2D(uMap, vUv).rgb;
    if (uObserverSilhouette > 0.5) {
      gl_FragColor = vec4(vec3(0.00015), 1.0);
      return;
    }
    vec3 n = normalize(vWorldNormal);
    vec3 l = normalize(uSunPosition - vWorldPosition);
    vec3 v = normalize(cameraPosition - vWorldPosition);
    float mu0 = max(dot(n, l), 0.0);
    float mu = max(dot(n, v), 0.0);
    float visibility = finiteSunVisibility(vWorldPosition);
    float direct = mu0 * visibility;

    vec3 color;
    if (uIsEarth > 0.5) {
      float blueExcess = albedo.b - max(albedo.r, albedo.g);
      float water = smoothstep(0.015, 0.11, blueExcess) * (1.0 - smoothstep(0.38, 0.58, dot(albedo, vec3(0.2126,0.7152,0.0722))));
      vec3 h = normalize(l + v);
      float spec = pow(max(dot(n, h), 0.0), 180.0) * water * visibility * step(0.0, dot(n,l));
      float twilight = exp(-pow(dot(n,l) / 0.035, 2.0));
      color = albedo * (uNightFloor + 1.18 * direct);
      color += vec3(1.0,0.36,0.08) * twilight * 0.018 * visibility;
      color += vec3(1.55,1.72,1.95) * spec * 1.6;
    } else {
      float lommel = mu0 / max(mu0 + mu, 0.035);
      float photometric = 0.72 * lommel + 0.28 * mu0;
      color = albedo * (uNightFloor + 1.45 * photometric * visibility);
      float eclipseDepth = pow(clamp(1.0 - visibility, 0.0, 1.0), 1.35);
      color += albedo * vec3(0.44, 0.055, 0.018) * eclipseDepth * uLunarRed * (0.35 + 0.65*mu);
    }
    color *= uExposure;
    gl_FragColor = vec4(color, 1.0);
  }
`;

function makeBodyMaterial(texture, isEarth) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uMap: { value: texture },
      uSunPosition: { value: new THREE.Vector3() },
      uOccluderPosition: { value: new THREE.Vector3() },
      uSunRadius: { value: CONFIG.radii.sun / UNIT_KM },
      uOccluderRadius: { value: 1.0 },
      uEclipseEnabled: { value: 0.0 },
      uIsEarth: { value: isEarth ? 1.0 : 0.0 },
      uLunarRed: { value: isEarth ? 0.0 : 1.0 },
      uExposure: { value: 1.0 },
      uNightFloor: { value: isEarth ? 0.003 : 0.006 },
      uObserverSilhouette: { value: 0.0 },
      uPhotometryMode: { value: 1.0 },
      uPhotometryLut: { value: photometryCorrectionTexture },
      uPhotometryLutRange: { value: photometryLutRange.clone() },
      uPhotometryLutSize: { value: photometryLutSize.clone() },
    },
    vertexShader: commonVertexShader,
    fragmentShader: bodyFragmentShader,
    side: THREE.FrontSide,
    transparent: false,
    opacity: 1.0,
    blending: THREE.NoBlending,
    depthWrite: true,
    depthTest: true,
    toneMapped: false,
  });
}

const earthGeometry = makeLatLonGeometry(CONFIG.mesh.earth_lat, CONFIG.mesh.earth_lon, [
  CONFIG.radii.earth_equatorial / UNIT_KM,
  CONFIG.radii.earth_equatorial / UNIT_KM,
  CONFIG.radii.earth_polar / UNIT_KM,
]);
const moonGeometry = makeLatLonGeometry(CONFIG.mesh.moon_lat, CONFIG.mesh.moon_lon, [
  CONFIG.radii.moon / UNIT_KM,
  CONFIG.radii.moon / UNIT_KM,
  CONFIG.radii.moon / UNIT_KM,
]);
const sunGeometry = makeLatLonGeometry(CONFIG.mesh.sun_lat, CONFIG.mesh.sun_lon, [
  CONFIG.radii.sun / UNIT_KM,
  CONFIG.radii.sun / UNIT_KM,
  CONFIG.radii.sun / UNIT_KM,
]);

const earthMaterial = makeBodyMaterial(earthTexture, true);
const moonMaterial = makeBodyMaterial(moonTexture, false);
const earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);
const moonMesh = new THREE.Mesh(moonGeometry, moonMaterial);
earthMesh.frustumCulled = false;
moonMesh.frustumCulled = false;
earthMesh.renderOrder = 0;
moonMesh.renderOrder = 0;
bodyGroup.add(earthMesh, moonMesh);

// A dedicated opaque-body silhouette pass protects Earth and Moon from
// screen-space bloom leaking across their discs.  Only the atmosphere is
// translucent; the physical bodies remain color- and depth-opaque.
const bodyMaskScene = new THREE.Scene();
bodyMaskScene.background = new THREE.Color(0x000000);
const bodyMaskMaterial = new THREE.MeshBasicMaterial({
  color: 0xffffff,
  side: THREE.FrontSide,
  transparent: false,
  depthWrite: true,
  depthTest: true,
  toneMapped: false,
});
const earthMaskMesh = new THREE.Mesh(earthGeometry, bodyMaskMaterial);
const moonMaskMesh = new THREE.Mesh(moonGeometry, bodyMaskMaterial);
earthMaskMesh.frustumCulled = false;
moonMaskMesh.frustumCulled = false;
bodyMaskScene.add(earthMaskMesh, moonMaskMesh);

const sunMaterial = new THREE.ShaderMaterial({
  uniforms: { uTime: { value: 0 }, uExposure: { value: 1.0 } },
  vertexShader: `
    #include <common>
    #include <logdepthbuf_pars_vertex>
    varying vec3 vNormalW;
    varying vec3 vLocal;
    varying vec3 vWorld;
    void main(){
      vLocal = normalize(position);
      vNormalW = normalize(normalMatrix * normal);
      vec4 wp = modelMatrix * vec4(position,1.0);
      vWorld = wp.xyz;
      gl_Position = projectionMatrix * viewMatrix * wp;
      #include <logdepthbuf_vertex>
    }
  `,
  fragmentShader: `
    precision highp float;
    #include <common>
    #include <logdepthbuf_pars_fragment>
    uniform float uTime;
    uniform float uExposure;
    varying vec3 vNormalW;
    varying vec3 vLocal;
    varying vec3 vWorld;
    float hash31(vec3 p){
      p = fract(p*0.1031); p += dot(p,p.yzx+33.33); return fract((p.x+p.y)*p.z);
    }
    float noise(vec3 p){
      vec3 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
      return mix(mix(mix(hash31(i+vec3(0,0,0)),hash31(i+vec3(1,0,0)),f.x),
                     mix(hash31(i+vec3(0,1,0)),hash31(i+vec3(1,1,0)),f.x),f.y),
                 mix(mix(hash31(i+vec3(0,0,1)),hash31(i+vec3(1,0,1)),f.x),
                     mix(hash31(i+vec3(0,1,1)),hash31(i+vec3(1,1,1)),f.x),f.y),f.z);
    }
    void main(){
      #include <logdepthbuf_fragment>
      vec3 v = normalize(cameraPosition - vWorld);
      float mu = clamp(dot(normalize(vNormalW), v), 0.0, 1.0);
      float limb = 0.46 + 0.54*pow(mu,0.42);
      vec3 p = vLocal*22.0 + vec3(uTime*0.008,0.0,uTime*0.003);
      float g = 0.58*noise(p)+0.28*noise(p*2.3)+0.14*noise(p*5.1);
      float cells = 0.78 + 0.38*g;
      vec3 hot = mix(vec3(4.8,1.15,0.12), vec3(10.5,6.4,2.35), clamp(cells,0.0,1.0));
      gl_FragColor = vec4(hot*limb*uExposure,1.0);
    }
  `,
  transparent: false,
  opacity: 1.0,
  blending: THREE.NoBlending,
  depthWrite: true,
  depthTest: true,
  toneMapped: false,
});
const sunMesh = new THREE.Mesh(sunGeometry, sunMaterial);
sunMesh.frustumCulled = false;
sunMesh.renderOrder = 0;
bodyGroup.add(sunMesh);

function makeAtmosphereMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      uSunPosition: { value: new THREE.Vector3() },
      uOccluderPosition: { value: new THREE.Vector3() },
      uSunRadius: { value: CONFIG.radii.sun / UNIT_KM },
      uOccluderRadius: { value: 1.0 },
      uEclipseEnabled: { value: 0.0 },
      uPhotometryMode: { value: 1.0 },
      uPhotometryLut: { value: photometryCorrectionTexture },
      uPhotometryLutRange: { value: photometryLutRange.clone() },
      uPhotometryLutSize: { value: photometryLutSize.clone() },
      uColor: { value: new THREE.Color(0.15, 0.48, 1.65) },
      uStrength: { value: 2.2 },
    },
    vertexShader: `
      #include <common>
      #include <logdepthbuf_pars_vertex>
      varying vec3 vNormalW; varying vec3 vWorld;
      void main(){
        vNormalW=normalize(normalMatrix*normal);
        vec4 wp=modelMatrix*vec4(position,1.0); vWorld=wp.xyz;
        gl_Position=projectionMatrix*viewMatrix*wp;
        #include <logdepthbuf_vertex>
      }
    `,
    fragmentShader: `
      precision highp float;
      #include <common>
      #include <logdepthbuf_pars_fragment>
      uniform vec3 uSunPosition; uniform vec3 uOccluderPosition;
      uniform float uSunRadius; uniform float uOccluderRadius; uniform float uEclipseEnabled;
      uniform float uPhotometryMode; uniform sampler2D uPhotometryLut;
      uniform vec4 uPhotometryLutRange; uniform vec2 uPhotometryLutSize;
      uniform vec3 uColor; uniform float uStrength;
      varying vec3 vNormalW; varying vec3 vWorld;
      float circleVisibility(float rOcc,float rSun,float d){
        if(d>=rOcc+rSun)return 1.0;
        if(d<=abs(rOcc-rSun))return rOcc>=rSun?0.0:clamp(1.0-rOcc*rOcc/max(rSun*rSun,1e-12),0.0,1.0);
        float ds=max(d,1e-9);
        float a1=clamp((ds*ds+rOcc*rOcc-rSun*rSun)/(2.0*ds*rOcc),-1.0,1.0);
        float a2=clamp((ds*ds+rSun*rSun-rOcc*rOcc)/(2.0*ds*rSun),-1.0,1.0);
        float term=max((-ds+rOcc+rSun)*(ds+rOcc-rSun)*(ds-rOcc+rSun)*(ds+rOcc+rSun),0.0);
        float overlap=rOcc*rOcc*acos(a1)+rSun*rSun*acos(a2)-0.5*sqrt(term);
        return clamp(1.0-overlap/max(3.141592653589793*rSun*rSun,1e-12),0.0,1.0);
      }
      float limbDarkenedVisibility(float rOcc,float rSun,float d){
        float uniformVisibility=circleVisibility(rOcc,rSun,d);
        if(d>=rOcc+rSun)return 1.0;if(rOcc>=rSun&&d<=rOcc-rSun)return 0.0;
        float q=rOcc/max(rSun,1.0e-9),ss=d/max(rSun,1.0e-9);
        if(q<0.0||ss<0.0||q>uPhotometryLutRange.x||ss>uPhotometryLutRange.y)return uniformVisibility;
        vec2 grid=vec2(ss/uPhotometryLutRange.y,q/uPhotometryLutRange.x);
        vec2 uv=(grid*(uPhotometryLutSize-vec2(1.0))+vec2(0.5))/uPhotometryLutSize;
        float encodedCorrection=texture2D(uPhotometryLut,uv).r;
        float correction=mix(uPhotometryLutRange.z,uPhotometryLutRange.w,encodedCorrection);
        return clamp(uniformVisibility+correction,0.0,1.0);
      }
      float finiteSunVisibility(vec3 p){
        if(uEclipseEnabled<0.5)return 1.0;
        vec3 s=uSunPosition-p, o=uOccluderPosition-p;
        float rs=asin(clamp(uSunRadius/max(length(s),uSunRadius),0.0,1.0));
        float ro=asin(clamp(uOccluderRadius/max(length(o),uOccluderRadius),0.0,1.0));
        float sep=acos(clamp(dot(normalize(s),normalize(o)),-1.0,1.0));
        return uPhotometryMode>0.5?limbDarkenedVisibility(ro,rs,sep):circleVisibility(ro,rs,sep);
      }
      void main(){
        #include <logdepthbuf_fragment>
        vec3 n=normalize(vNormalW); vec3 v=normalize(cameraPosition-vWorld); vec3 l=normalize(uSunPosition-vWorld);
        float rim=pow(max(1.0-abs(dot(n,v)),0.0),5.2);
        float visibility=finiteSunVisibility(vWorld);
        float day=(0.08+0.92*smoothstep(-0.18,0.35,dot(n,l)))*visibility;
        float alpha=clamp(rim*day*0.38,0.0,0.46);
        gl_FragColor=vec4(uColor*uStrength*rim*day,alpha);
      }
    `,
    side: THREE.BackSide,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    depthTest: true,
  });
}
const atmosphereGeometry = makeLatLonGeometry(Math.max(64, Math.floor(CONFIG.mesh.earth_lat/2)), Math.max(128, Math.floor(CONFIG.mesh.earth_lon/2)), [
  CONFIG.radii.earth_equatorial/UNIT_KM*1.018,
  CONFIG.radii.earth_equatorial/UNIT_KM*1.018,
  CONFIG.radii.earth_polar/UNIT_KM*1.022,
]);
const atmosphereMaterial = makeAtmosphereMaterial();
const atmosphereMesh = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
atmosphereMesh.frustumCulled = false;
atmosphereMesh.renderOrder = 30;
atmosphereGroup.add(atmosphereMesh);

function makeRadialTexture(size=512) {
  const c=document.createElement('canvas'); c.width=c.height=size;
  const ctx=c.getContext('2d'); const g=ctx.createRadialGradient(size/2,size/2,0,size/2,size/2,size/2);
  g.addColorStop(0.0,'rgba(255,255,245,1)');
  g.addColorStop(0.07,'rgba(255,255,250,.94)');
  g.addColorStop(0.20,'rgba(255,252,232,.46)');
  g.addColorStop(0.48,'rgba(220,235,255,.12)');
  g.addColorStop(1.0,'rgba(220,235,255,0)');
  ctx.fillStyle=g; ctx.fillRect(0,0,size,size);
  const t=new THREE.CanvasTexture(c); t.colorSpace=THREE.SRGBColorSpace; return t;
}
const corona = new THREE.Sprite(new THREE.SpriteMaterial({
  map: makeRadialTexture(), transparent:true, blending:THREE.AdditiveBlending,
  depthWrite:false, depthTest:true, color:new THREE.Color(2.35,1.55,0.78), opacity:0.92,
}));
const coronaScale = CONFIG.radii.sun/UNIT_KM*4.4;
corona.scale.set(coronaScale,coronaScale,1);
corona.renderOrder=40;
bodyGroup.add(corona);

function makeStars(count=4200) {
  const rng = mulberry32(424242);
  const p = new Float32Array(count*3);
  const col = new Float32Array(count*3);
  const r = 260000;
  for(let i=0;i<count;i++){
    let x,y,z,n;
    do { x=rng()*2-1; y=rng()*2-1; z=rng()*2-1; n=Math.hypot(x,y,z); } while(n<1e-4);
    const rr=r*(0.76+0.24*rng());
    p[i*3]=x/n*rr; p[i*3+1]=y/n*rr; p[i*3+2]=z/n*rr;
    const k=0.55+1.8*Math.pow(rng(),7);
    col[i*3]=k*(0.82+0.18*rng()); col[i*3+1]=k*(0.86+0.14*rng()); col[i*3+2]=k;
  }
  const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.BufferAttribute(p,3)); g.setAttribute('color',new THREE.BufferAttribute(col,3));
  const m=new THREE.PointsMaterial({size:1.35,sizeAttenuation:false,vertexColors:true,transparent:true,opacity:.9,depthWrite:false});
  return new THREE.Points(g,m);
}
function mulberry32(a){return function(){let t=a+=0x6D2B79F5;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;}}
const stars=makeStars(); stars.renderOrder=-20; scene.add(stars);

function makeBeamMaterial(opacity, colorScale) {
  const m = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    depthTest: true,
    vertexColors: true,
  });
  m.toneMapped=false;
  m.userData.colorScale=colorScale;
  return m;
}
const beamGeometry = new THREE.CylinderGeometry(1,1,1,10,1,true);
const yAxis = new THREE.Vector3(0,1,0);
function makeBeamSet(count, coreRadius, haloRadius) {
  const core = new THREE.InstancedMesh(beamGeometry, makeBeamMaterial(0.82,1), count);
  const halo = new THREE.InstancedMesh(beamGeometry, makeBeamMaterial(0.13,1), count);
  core.instanceMatrix.setUsage(THREE.DynamicDrawUsage); halo.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  core.frustumCulled=false; halo.frustumCulled=false;
  core.renderOrder=20; halo.renderOrder=19;
  core.userData.radius=coreRadius; halo.userData.radius=haloRadius;
  return {core,halo,count};
}
function updateBeamSet(set, paths, palette) {
  const dummy=new THREE.Object3D(),dir=new THREE.Vector3(),mid=new THREE.Vector3(),color=new THREE.Color();
  const hidden=new THREE.Matrix4().makeScale(0,0,0);
  const meanWeight=1/Math.max(1,paths.length);
  for(let i=0;i<set.count;i++){
    if(i>=paths.length){set.core.setMatrixAt(i,hidden);set.halo.setMatrixAt(i,hidden);continue;}
    const p=paths[i],a=new THREE.Vector3(...p[0]),b=new THREE.Vector3(...p[1]);
    dir.subVectors(b,a);const len=dir.length();mid.addVectors(a,b).multiplyScalar(.5);
    const weight=Math.max(1e-6,Number(p[3]??meanWeight));
    const weightScale=THREE.MathUtils.clamp(Math.sqrt(weight/meanWeight),0.45,1.75);
    dummy.position.copy(mid);dummy.quaternion.setFromUnitVectors(yAxis,dir.clone().normalize());
    dummy.scale.set(set.core.userData.radius*weightScale,len,set.core.userData.radius*weightScale);dummy.updateMatrix();set.core.setMatrixAt(i,dummy.matrix);
    dummy.scale.set(set.halo.userData.radius*weightScale,len,set.halo.userData.radius*weightScale);dummy.updateMatrix();set.halo.setMatrixAt(i,dummy.matrix);
    const rgb=p[2]?palette.blocked:palette.reached;
    const intensity=THREE.MathUtils.clamp(.66+.34*weightScale,.55,1.25);
    color.setRGB(rgb[0]*intensity,rgb[1]*intensity,rgb[2]*intensity);set.core.setColorAt(i,color);
    color.setRGB(rgb[0]*.62*intensity,rgb[1]*.40*intensity,rgb[2]*.16*intensity);set.halo.setColorAt(i,color);
  }
  set.core.instanceMatrix.needsUpdate=true;set.halo.instanceMatrix.needsUpdate=true;
  if(set.core.instanceColor)set.core.instanceColor.needsUpdate=true;
  if(set.halo.instanceColor)set.halo.instanceColor.needsUpdate=true;
}

const rayCount=CONFIG.ray_count;
const localBeams=makeBeamSet(rayCount,0.035,0.18);
// The one-AU view needs display-thickened shafts: their centerlines and
// endpoints are exact, while the 12,000/70,000 km core/halo radii are a
// screen-readability aid disclosed in the UI.
const fullBeams=makeBeamSet(rayCount,12.0,70.0);
function makeDotTexture(size=128){
  const c=document.createElement('canvas');c.width=c.height=size;const x=c.getContext('2d');
  const g=x.createRadialGradient(size/2,size/2,0,size/2,size/2,size/2);
  g.addColorStop(0,'rgba(255,255,255,1)');g.addColorStop(.16,'rgba(255,238,170,.95)');
  g.addColorStop(.48,'rgba(255,154,50,.28)');g.addColorStop(1,'rgba(255,100,20,0)');
  x.fillStyle=g;x.fillRect(0,0,size,size);return new THREE.CanvasTexture(c);
}
const beamDotTexture=makeDotTexture();
function makeBeamPoints(maxCount,size){
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(maxCount*2*3),3));
  geometry.setAttribute('color',new THREE.BufferAttribute(new Float32Array(maxCount*2*3),3));
  geometry.setDrawRange(0,0);
  const material=new THREE.PointsMaterial({map:beamDotTexture,size,sizeAttenuation:false,transparent:true,opacity:.95,vertexColors:true,blending:THREE.AdditiveBlending,depthWrite:false,depthTest:true,alphaTest:.015});
  const points=new THREE.Points(geometry,material);points.frustumCulled=false;return points;
}
function updateBeamPoints(points,paths,palette){
  const pos=points.geometry.attributes.position.array,col=points.geometry.attributes.color.array;
  const meanWeight=1/Math.max(1,paths.length);let n=0;
  for(const path of paths){
    const rgb=path[2]?palette.blocked:palette.reached;
    const weightScale=THREE.MathUtils.clamp(Math.sqrt(Math.max(1e-6,Number(path[3]??meanWeight))/meanWeight),.45,1.75);
    for(let e=0;e<2;e++){
      const p=path[e];pos[n*3]=p[0];pos[n*3+1]=p[1];pos[n*3+2]=p[2];
      col[n*3]=rgb[0]*weightScale;col[n*3+1]=rgb[1]*weightScale;col[n*3+2]=rgb[2]*weightScale;n++;
    }
  }
  points.geometry.setDrawRange(0,n);points.geometry.attributes.position.needsUpdate=true;points.geometry.attributes.color.needsUpdate=true;
}
const localBeamPoints=makeBeamPoints(rayCount,7.0);
const fullBeamPoints=makeBeamPoints(rayCount,5.0);
localRayGroup.add(localBeams.halo,localBeams.core,localBeamPoints);
fullRayGroup.add(fullBeams.halo,fullBeams.core,fullBeamPoints);

function makeDiagnosticBeam(color, opacity, radius, haloRadius, dashed=false) {
  const core=makeBeamSet(1,radius,haloRadius);
  const c=new THREE.Color(color);
  core.core.material.color.copy(c); core.core.material.vertexColors=false; core.core.material.opacity=opacity;
  core.halo.material.color.copy(c); core.halo.material.vertexColors=false; core.halo.material.opacity=opacity*0.15;
  core.dashed=dashed;
  return core;
}
const centralBeam=makeDiagnosticBeam(0xffe26e,.95,.028,.13);
const shadowBeam=makeDiagnosticBeam(0xe47aff,.70,.022,.10,true);
const umbraUpper=makeDiagnosticBeam(0xff665c,.72,.018,.075);
const umbraLower=makeDiagnosticBeam(0xff665c,.72,.018,.075);
const penUpper=makeDiagnosticBeam(0x6fc8ff,.62,.015,.065);
const penLower=makeDiagnosticBeam(0x6fc8ff,.62,.015,.065);
for(const b of [centralBeam,shadowBeam,umbraUpper,umbraLower,penUpper,penLower]) boundaryGroup.add(b.halo,b.core);

function updateSingleBeam(set, path) {
  updateBeamSet(set, path && path.length===2 ? [[path[0],path[1],false]] : [], {reached:[2.8,2.2,.72],blocked:[2.8,2.2,.72]});
}

function makeAxisGrid() {
  const group=new THREE.Group();
  const matX=new THREE.LineBasicMaterial({color:0xff586a,transparent:true,opacity:.5});
  const matY=new THREE.LineBasicMaterial({color:0x5fffa2,transparent:true,opacity:.5});
  const matZ=new THREE.LineBasicMaterial({color:0x65a9ff,transparent:true,opacity:.5});
  const length=60;
  function line(a,b,m){const g=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a),new THREE.Vector3(...b)]);group.add(new THREE.Line(g,m));}
  line([-length,0,0],[length,0,0],matX); line([0,-length,0],[0,length,0],matY); line([0,0,-length],[0,0,length],matZ);
  return group;
}
const axes=makeAxisGrid(); scientificGroup.add(axes); scientificGroup.visible=false;

function makeOrbitOverlays(){
  const definitions=Array.isArray(CONFIG.orbit_overlays)?CONFIG.orbit_overlays:[];
  return definitions.map((definition)=>{
    if(!definition||!definition.enabled||!Array.isArray(definition.path)||definition.path.length<2){
      return {line:null,marker:null};
    }
    const points=definition.path.map((point)=>new THREE.Vector3(...point));
    const geometry=new THREE.BufferGeometry().setFromPoints(points);
    const material=new THREE.LineBasicMaterial({
      color:definition.color||0x5fffa2,transparent:true,opacity:.72,
      depthWrite:false,depthTest:true,
    });
    const line=new THREE.Line(geometry,material);line.renderOrder=12;
    const marker=new THREE.Mesh(
      new THREE.SphereGeometry(.17,18,10),
      new THREE.MeshBasicMaterial({color:definition.color||0x5fffa2,toneMapped:false,depthWrite:false,depthTest:true}),
    );
    marker.renderOrder=26;orbitGroup.add(line,marker);
    return {line,marker};
  });
}
const orbitOverlays=makeOrbitOverlays();

function makeLabel(text, color='#ffffff') {
  const c=document.createElement('canvas');c.width=1024;c.height=160;const ctx=c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);ctx.font='600 44px system-ui, sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
  ctx.shadowColor='rgba(0,0,0,.92)';ctx.shadowBlur=14;ctx.lineWidth=8;ctx.strokeStyle='rgba(0,0,0,.90)';ctx.strokeText(text,512,80);ctx.fillStyle=color;ctx.fillText(text,512,80);
  const t=new THREE.CanvasTexture(c);t.colorSpace=THREE.SRGBColorSpace;
  const material=new THREE.SpriteMaterial({map:t,transparent:true,depthTest:false,depthWrite:false});
  const sprite=new THREE.Sprite(material);sprite.renderOrder=1000;sprite.userData.baseAspect=6.4;return sprite;
}
const earthLabel=makeLabel(CONFIG.labels.earth,'#a7d7ff');
const moonLabel=makeLabel(CONFIG.labels.moon,'#e6e6e6');
const sunLabel=makeLabel(CONFIG.labels.sun,'#ffe37b');
const targetLabel=makeLabel(CONFIG.labels.target,'#ffd36d');
labelGroup.add(earthLabel,moonLabel,sunLabel,targetLabel);
const targetMarker=new THREE.Mesh(
  new THREE.SphereGeometry(.11,24,12),
  new THREE.MeshBasicMaterial({color:0xffd36d,transparent:false,depthTest:true,depthWrite:true,toneMapped:false})
);
targetMarker.renderOrder=25;labelGroup.add(targetMarker);

let currentIndex=0;
let elapsedSeconds=0;
let currentRenderState=null;
const eventStartJD=Number(CONFIG.states[0].jd_utc);
const eventDuration=Math.max(0,Number(CONFIG.event_duration_seconds||0));

function rowsToQuaternion(rows){
  const m=new THREE.Matrix4();
  m.set(rows[0][0],rows[0][1],rows[0][2],0,rows[1][0],rows[1][1],rows[1][2],0,rows[2][0],rows[2][1],rows[2][2],0,0,0,0,1);
  return new THREE.Quaternion().setFromRotationMatrix(m).normalize();
}
function setObjectPose(object,quaternion,position){
  object.matrixAutoUpdate=true;object.position.set(...position);object.quaternion.copy(quaternion);object.scale.set(1,1,1);object.updateMatrix();
}
function lerpArray(a,b,t){return a.map((v,i)=>v+(b[i]-v)*t);}
function lerpNumber(a,b,t){return Number(a)+(Number(b)-Number(a))*t;}
function interpObject(a,b,t){
  if(!a)return b||null;if(!b)return a;
  const out={...a};
  for(const key of ['position','up','north','east','sky_direction']) if(a[key]&&b[key])out[key]=lerpArray(a[key],b[key],t);
  for(const key of ['latitude_deg','longitude_east_deg','photosphere_visible','sun_altitude_deg','sun_azimuth_deg','moon_altitude_deg','moon_azimuth_deg']){
    if(Number.isFinite(Number(a[key]))&&Number.isFinite(Number(b[key])))out[key]=lerpNumber(a[key],b[key],t);
  }
  return out;
}
function bracketForJD(jd){
  const states=CONFIG.states;
  if(states.length===1||jd<=states[0].jd_utc)return [0,0,0];
  const last=states.length-1;if(jd>=states[last].jd_utc)return [last,last,0];
  let lo=0,hi=last;
  while(hi-lo>1){const mid=(lo+hi)>>1;if(states[mid].jd_utc<=jd)lo=mid;else hi=mid;}
  const span=Math.max(1e-15,states[hi].jd_utc-states[lo].jd_utc);
  return [lo,hi,THREE.MathUtils.clamp((jd-states[lo].jd_utc)/span,0,1)];
}
function renderStateAtElapsed(seconds){
  const clamped=THREE.MathUtils.clamp(Number(seconds)||0,0,eventDuration);
  const jd=eventStartJD+clamped/86400;
  const [i0,i1,t]=bracketForJD(jd),a=CONFIG.states[i0],b=CONFIG.states[i1];
  const qa=rowsToQuaternion(a.earth_rotation),qb=rowsToQuaternion(b.earth_rotation);
  const ma=rowsToQuaternion(a.moon_rotation),mb=rowsToQuaternion(b.moon_rotation);
  const nearest=(t<.5?a:b),nearestIndex=(t<.5?i0:i1);
  const orbitA=Array.isArray(a.ssapy_orbit_positions)?a.ssapy_orbit_positions:[];
  const orbitB=Array.isArray(b.ssapy_orbit_positions)?b.ssapy_orbit_positions:orbitA;
  const orbitPositions=orbitA.map((position,index)=>{
    const other=orbitB[index]||position;
    return position&&other?lerpArray(position,other,t):(position||other||null);
  });
  return {
    jd_utc:jd,phase:nearest.phase,time:nearest.time,nearest,nearestIndex,
    moon:lerpArray(a.moon,b.moon,t),sun:lerpArray(a.sun,b.sun,t),
    earthQuat:new THREE.Quaternion().slerpQuaternions(qa,qb,t).normalize(),
    moonQuat:new THREE.Quaternion().slerpQuaternions(ma,mb,t).normalize(),
    earth_moon_km:lerpNumber(a.earth_moon_km,b.earth_moon_km,t),
    sun_earth_km:lerpNumber(a.sun_earth_km,b.sun_earth_km,t),
    target:interpObject(a.target,b.target,t),observer:interpObject(a.observer,b.observer,t),
    orbitPositions,
  };
}
function dateStringFromJD(jd){
  const ms=(Number(jd)-2440587.5)*86400000;
  const d=new Date(ms);return d.toISOString().replace('T',' ').replace('.000Z',' UTC');
}
function updateUniforms(state){
  const sun=new THREE.Vector3(...state.sun),moon=new THREE.Vector3(...state.moon);
  earthMaterial.uniforms.uSunPosition.value.copy(sun);moonMaterial.uniforms.uSunPosition.value.copy(sun);atmosphereMaterial.uniforms.uSunPosition.value.copy(sun);
  if(CONFIG.mode==='solar'){
    earthMaterial.uniforms.uEclipseEnabled.value=1;earthMaterial.uniforms.uOccluderPosition.value.copy(moon);earthMaterial.uniforms.uOccluderRadius.value=CONFIG.radii.moon/UNIT_KM;
    atmosphereMaterial.uniforms.uEclipseEnabled.value=1;atmosphereMaterial.uniforms.uOccluderPosition.value.copy(moon);atmosphereMaterial.uniforms.uOccluderRadius.value=CONFIG.radii.moon/UNIT_KM;
    moonMaterial.uniforms.uEclipseEnabled.value=0;
  }else{
    earthMaterial.uniforms.uEclipseEnabled.value=0;atmosphereMaterial.uniforms.uEclipseEnabled.value=0;moonMaterial.uniforms.uEclipseEnabled.value=1;
    moonMaterial.uniforms.uOccluderPosition.value.set(0,0,0);moonMaterial.uniforms.uOccluderRadius.value=CONFIG.radii.earth_shadow/UNIT_KM;
  }
}
function updateLabelPositions(state){
  const earthNorth=new THREE.Vector3(0,0,1).applyQuaternion(state.earthQuat).normalize();
  const moonNorth=new THREE.Vector3(0,0,1).applyQuaternion(state.moonQuat).normalize();
  const moon=new THREE.Vector3(...state.moon),sun=new THREE.Vector3(...state.sun),targetPoint=new THREE.Vector3(...state.target.position);
  earthLabel.position.copy(earthNorth).multiplyScalar(CONFIG.radii.earth_polar/UNIT_KM*1.58);
  moonLabel.position.copy(moon).addScaledVector(moonNorth,CONFIG.radii.moon/UNIT_KM*4.1);
  const sunUp=(activePreset==='sun'?camera.up:new THREE.Vector3(0,0,1)).clone().normalize();
  sunLabel.position.copy(sun).addScaledVector(sunUp,CONFIG.radii.sun/UNIT_KM*1.35);
  targetMarker.position.copy(targetPoint);
  const targetOffset=targetPoint.length()>1e-6?targetPoint.clone().normalize():earthNorth;
  if(CONFIG.mode==='lunar')targetOffset.copy(moonNorth);
  targetLabel.position.copy(targetPoint).addScaledVector(targetOffset,CONFIG.mode==='solar'?.45:CONFIG.radii.moon/UNIT_KM*4.8);
}
function updateHud(state){
  const m=state.nearest.ray_metrics||{};
  document.getElementById('metricBackend').textContent=CONFIG.backend;
  document.getElementById('metricFrame').textContent=CONFIG.frame_backend;
  document.getElementById('metricScope').textContent=CONFIG.scope_label;
  document.getElementById('metricDistance').textContent=state.earth_moon_km.toLocaleString(undefined,{maximumFractionDigits:1})+' km';
  document.getElementById('metricRays').textContent=`${m.count??CONFIG.ray_count} weighted full-disc samples`;
  const photometryLabel=(photometrySelect&&photometrySelect.value==='uniform')?'uniform-disc geometry':(CONFIG.photometry?.name||'quadratic-visible');
  const photometryMetric=document.getElementById('metricPhotometry');if(photometryMetric)photometryMetric.textContent=photometryLabel;
  document.getElementById('metricBlocked').textContent=((m.blocked_weight??0)*100).toFixed(2)+'% quadrature weight';
  document.getElementById('metricTarget').textContent=state.target.label||'eclipse evaluation point';
  document.getElementById('metricVisibility').textContent=(Number(state.target.photosphere_visible||0)*100).toFixed(4)+'%';
  const irradianceMetric=document.getElementById('metricIrradiance');if(irradianceMetric)irradianceMetric.textContent=(Number(state.target.irradiance_visible??state.target.photosphere_visible??0)*100).toFixed(4)+'%';
  document.getElementById('metricLocation').textContent=(Number.isFinite(Number(state.target.latitude_deg))?`${Number(state.target.latitude_deg).toFixed(3)}°, ${Number(state.target.longitude_east_deg).toFixed(3)}° E`:'not a terrestrial surface point');
  document.getElementById('metricObserver').textContent=state.observer?`Sun ${Number(state.observer.sun_altitude_deg).toFixed(2)}° alt · Moon ${Number(state.observer.moon_altitude_deg).toFixed(2)}° alt`:'space-system evaluation';
}
function applyElapsed(seconds){
  elapsedSeconds=THREE.MathUtils.clamp(Number(seconds)||0,0,eventDuration);
  const state=renderStateAtElapsed(elapsedSeconds);currentRenderState=state;currentIndex=state.nearestIndex;
  setObjectPose(earthMesh,state.earthQuat,[0,0,0]);setObjectPose(earthMaskMesh,state.earthQuat,[0,0,0]);setObjectPose(atmosphereMesh,state.earthQuat,[0,0,0]);
  setObjectPose(moonMesh,state.moonQuat,state.moon);setObjectPose(moonMaskMesh,state.moonQuat,state.moon);
  sunMesh.position.set(...state.sun);sunMesh.matrixAutoUpdate=true;corona.position.set(...state.sun);updateUniforms(state);
  orbitOverlays.forEach((overlay,index)=>{
    const position=state.orbitPositions[index];
    if(overlay.marker&&position)overlay.marker.position.set(...position);
  });
  const near=state.nearest;
  updateBeamSet(localBeams,near.local_rays,{reached:[4.6,3.8,1.45],blocked:[4.8,2.15,.52]});
  updateBeamSet(fullBeams,near.full_rays,{reached:[4.6,3.8,1.45],blocked:[4.8,2.15,.52]});
  updateBeamPoints(localBeamPoints,near.local_rays,{reached:[4.6,3.8,1.45],blocked:[4.8,2.15,.52]});
  updateBeamPoints(fullBeamPoints,near.full_rays,{reached:[4.6,3.8,1.45],blocked:[4.8,2.15,.52]});
  updateSingleBeam(centralBeam,near.central);updateSingleBeam(shadowBeam,near.shadow);
  updateSingleBeam(umbraUpper,near.umbra[0]);updateSingleBeam(umbraLower,near.umbra[1]);updateSingleBeam(penUpper,near.penumbra[0]);updateSingleBeam(penLower,near.penumbra[1]);
  updateLabelPositions(state);timeEl.textContent=dateStringFromJD(state.jd_utc);phaseEl.textContent=state.phase;slider.value=String(elapsedSeconds);updateHud(state);
  updateCamera(true);
}

function updateLabelScales(){
  const h=Math.max(1,renderer.domElement.clientHeight),fov=Math.tan(THREE.MathUtils.degToRad(camera.fov*.5));
  for(const label of [earthLabel,moonLabel,sunLabel,targetLabel]){
    const d=Math.max(.01,camera.position.distanceTo(label.position)),worldPerPixel=2*d*fov/h,height=worldPerPixel*28;
    label.scale.set(height*(label.userData.baseAspect||6.4),height,1);
  }
  const markerScale=Math.max(.025,camera.position.distanceTo(targetMarker.position)*.00018);targetMarker.scale.setScalar(markerScale);
}
function setGroupVisibility(){
  const observerView=activePreset==='observer';
  localRayGroup.visible=rayToggle.checked&&activePreset!=='true'&&!observerView;
  fullRayGroup.visible=rayToggle.checked&&activePreset==='true';boundaryGroup.visible=boundaryToggle.checked&&!observerView;
  atmosphereGroup.visible=atmosphereToggle.checked;stars.visible=starToggle.checked;labelGroup.visible=labelsToggle.checked&&!observerView;scientificGroup.visible=scientificToggle.checked&&!observerView;
  orbitGroup.visible=Boolean(orbitToggle&&orbitToggle.checked&&!observerView&&orbitOverlays.some((overlay)=>overlay.line));
  moonMaterial.uniforms.uObserverSilhouette.value=observerView?1.0:0.0;
  if(observerView)corona.material.color.setRGB(2.8,2.9,3.15);else corona.material.color.setRGB(2.35,1.55,.78);
}
function applyPhotometryMode(){
  const useLimb=!photometrySelect||photometrySelect.value!=='uniform';
  for(const material of [earthMaterial,moonMaterial]){
    material.uniforms.uPhotometryMode.value=useLimb?1.0:0.0;
  }
  atmosphereMaterial.uniforms.uPhotometryMode.value=useLimb?1.0:0.0;
  if(currentRenderState)updateHud(currentRenderState);
}
if(photometrySelect){
  photometrySelect.value=(CONFIG.photometry&&CONFIG.photometry.kind==='uniform')?'uniform':'limb';
  photometrySelect.addEventListener('change',applyPhotometryMode);
}
for(const el of [rayToggle,boundaryToggle,atmosphereToggle,starToggle,labelsToggle,scientificToggle,orbitToggle].filter(Boolean))el.addEventListener('change',setGroupVisibility);

const reducedMotion=Boolean(window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches);
let playing=Boolean(CONFIG.animate&&CONFIG.states.length>1&&eventDuration>0&&!reducedMotion);
playButton.textContent=playing?'Pause':'Play';
playButton.addEventListener('click',()=>{playing=!playing;playButton.textContent=playing?'Pause':'Play';});
slider.max=String(eventDuration);slider.step=eventDuration>0?'0.1':'1';
slider.addEventListener('input',()=>{playing=false;playButton.textContent='Play';applyElapsed(Number(slider.value));});

let target=new THREE.Vector3(),panOffset=new THREE.Vector3();
let spherical={radius:520,theta:-1.55,phi:1.15};
let activePreset='system';
let lastCameraUp=new THREE.Vector3(0,0,1);
function bodyAxisFromQuaternion(q,axisIndex=2){
  const axis=axisIndex===0?new THREE.Vector3(1,0,0):axisIndex===1?new THREE.Vector3(0,1,0):new THREE.Vector3(0,0,1);
  return axis.applyQuaternion(q).normalize();
}
function presetBaseTarget(state,name){
  const moon=new THREE.Vector3(...state.moon),sun=new THREE.Vector3(...state.sun);
  if(name==='earth')return new THREE.Vector3();
  if(name==='moon')return moon;
  if(name==='optics'||name==='system')return moon.clone().multiplyScalar(.5);
  if(name==='true')return sun.clone().multiplyScalar(.5);
  if(name==='sun')return sun;
  if(name==='observer'&&state.observer)return new THREE.Vector3(...state.observer.position).add(new THREE.Vector3(...state.observer.sky_direction).multiplyScalar(1000));
  return new THREE.Vector3();
}
function preferredCameraUp(state){
  let preferred;
  if(activePreset==='moon')preferred=bodyAxisFromQuaternion(state.moonQuat,2);
  else if(activePreset==='observer'&&state.observer)preferred=new THREE.Vector3(...state.observer.north).normalize();
  else if(activePreset==='optics'||activePreset==='sun')preferred=new THREE.Vector3(0,0,1);
  else preferred=bodyAxisFromQuaternion(state.earthQuat,2);
  const forward=target.clone().sub(camera.position).normalize();preferred.addScaledVector(forward,-preferred.dot(forward));
  if(preferred.lengthSq()<1e-10)preferred.set(0,0,1).addScaledVector(forward,-forward.z);
  if(preferred.lengthSq()<1e-10)preferred.set(0,1,0).addScaledVector(forward,-forward.y);
  preferred.normalize();if(preferred.dot(lastCameraUp)<0)preferred.multiplyScalar(-1);lastCameraUp.lerp(preferred,.24).normalize();return lastCameraUp;
}
function updateCamera(trackTarget=false){
  if(!currentRenderState)return;
  // The local observer is a real moving WGS-84 site, not merely a target
  // point.  Transport both the camera origin and sightline with that site;
  // otherwise Earth rotation leaves the camera behind in inertial space and
  // creates false Sun-Moon parallax during C1-C4 playback.
  if(trackTarget&&activePreset==='observer'&&currentRenderState.observer){
    const observer=currentRenderState.observer;
    const position=new THREE.Vector3(...observer.position);
    const up=new THREE.Vector3(...observer.up).normalize();
    const north=new THREE.Vector3(...observer.north).normalize();
    const sky=new THREE.Vector3(...observer.sky_direction).normalize();
    camera.position.copy(position).addScaledVector(up,.012);
    target.copy(position).addScaledVector(sky,1000);
    camera.up.copy(north);lastCameraUp.copy(north);camera.lookAt(target);
    return;
  }
  if(trackTarget)target.copy(presetBaseTarget(currentRenderState,activePreset)).add(panOffset);
  const sp=Math.sin(spherical.phi),cp=Math.cos(spherical.phi);
  camera.position.set(target.x+spherical.radius*sp*Math.cos(spherical.theta),target.y+spherical.radius*sp*Math.sin(spherical.theta),target.z+spherical.radius*cp);
  camera.up.copy(preferredCameraUp(currentRenderState));camera.lookAt(target);
}
function setCamera(position,targetValue,presetName,fov=38){
  const p=new THREE.Vector3(...position);activePreset=presetName;panOffset.set(0,0,0);target.set(...targetValue);const d=p.clone().sub(target);
  spherical.radius=Math.max(.02,d.length());spherical.theta=Math.atan2(d.y,d.x);spherical.phi=Math.acos(THREE.MathUtils.clamp(d.z/spherical.radius,-1,1));
  camera.fov=fov;camera.updateProjectionMatrix();lastCameraUp.set(0,0,1);updateCamera(true);setGroupVisibility();
}
function systemViewPose(state){
  const moon=new THREE.Vector3(...state.moon);
  const line=moon.clone().normalize();
  const north=bodyAxisFromQuaternion(state.earthQuat,2);
  let view=new THREE.Vector3().crossVectors(line,north);
  if(view.lengthSq()<1e-10)view.crossVectors(line,new THREE.Vector3(0,1,0));
  if(view.lengthSq()<1e-10)view.crossVectors(line,new THREE.Vector3(1,0,0));
  view.normalize();
  const sunDir=new THREE.Vector3(...state.sun).normalize();
  if(view.dot(sunDir)<0)view.multiplyScalar(-1);
  const mid=moon.clone().multiplyScalar(.5);
  const distance=Math.max(780,moon.length()*2.15);
  return {position:mid.clone().addScaledVector(view,distance),target:mid};
}
function preset(name){
  const s=currentRenderState||renderStateAtElapsed(elapsedSeconds),moon=new THREE.Vector3(...s.moon),sun=new THREE.Vector3(...s.sun);
  if(name==='earth')setCamera([34,-38,24],[0,0,0],'earth',35);
  else if(name==='moon')setCamera([moon.x+12,moon.y-15,moon.z+9],moon.toArray(),'moon',32);
  else if(name==='optics'){const mid=moon.clone().multiplyScalar(.5);setCamera([mid.x,mid.y-620,mid.z+8],mid.toArray(),'optics',30);}
  else if(name==='true'){const mid=sun.clone().multiplyScalar(.5);setCamera([mid.x,mid.y-92000,mid.z+32000],mid.toArray(),'true',32);}
  else if(name==='sun')setCamera([sun.x+2300,sun.y-2700,sun.z+1700],sun.toArray(),'sun',31);
  else if(name==='observer'&&s.observer){
    const pos=new THREE.Vector3(...s.observer.position),up=new THREE.Vector3(...s.observer.up).normalize(),sky=new THREE.Vector3(...s.observer.sky_direction).normalize();
    const eye=pos.clone().addScaledVector(up,.012),look=eye.clone().addScaledVector(sky,1000);setCamera(eye.toArray(),look.toArray(),'observer',3.2);
  }else{const pose=systemViewPose(s);setCamera(pose.position.toArray(),pose.target.toArray(),'system',36);}
}
document.querySelectorAll('[data-camera]').forEach(b=>b.addEventListener('click',()=>preset(b.dataset.camera)));

let drag=false,button=0,lastX=0,lastY=0;
canvas.addEventListener('pointerdown',e=>{drag=true;button=e.button;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointerup',e=>{drag=false;canvas.releasePointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{
  if(!drag)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;lastX=e.clientX;lastY=e.clientY;
  if(button===0){spherical.theta-=dx*0.005;spherical.phi=THREE.MathUtils.clamp(spherical.phi+dy*0.005,0.02,Math.PI-0.02);}
  else{const forward=target.clone().sub(camera.position).normalize();const right=new THREE.Vector3().crossVectors(forward,camera.up).normalize();const up=new THREE.Vector3().crossVectors(right,forward).normalize();const scale=spherical.radius*0.0015;panOffset.addScaledVector(right,-dx*scale).addScaledVector(up,dy*scale);target.copy(presetBaseTarget(currentRenderState,activePreset)).add(panOffset);}
  updateCamera();
});
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('wheel',e=>{e.preventDefault();spherical.radius*=Math.exp(e.deltaY*0.001);spherical.radius=THREE.MathUtils.clamp(spherical.radius,.03,400000);updateCamera();},{passive:false});

// Lightweight HDR bloom: bright extraction, two Gaussian scales, and an
// ACES-style composite. It is implemented here so the packaged output has no
// runtime CDN or JavaScript dependency beyond the bundled Three.js modules.
let sceneRT,bodyMaskRT,brightRT,blurA,blurB,smallA,smallB;
const postScene=new THREE.Scene();const postCamera=new THREE.OrthographicCamera(-1,1,1,-1,0,1);const postQuad=new THREE.Mesh(new THREE.PlaneGeometry(2,2));postScene.add(postQuad);
const brightMat=new THREE.ShaderMaterial({uniforms:{tInput:{value:null},threshold:{value:1.05},knee:{value:.55}},vertexShader:`varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}`,fragmentShader:`precision highp float;uniform sampler2D tInput;uniform float threshold;uniform float knee;varying vec2 vUv;void main(){vec3 c=texture2D(tInput,vUv).rgb;float b=max(max(c.r,c.g),c.b);float soft=clamp((b-threshold+knee)/(2.0*knee),0.0,1.0);soft=soft*soft*(3.0-2.0*soft);float w=max(b-threshold,0.0)+soft*knee;gl_FragColor=vec4(c*w/max(b,1e-5),1.0);}`});
const blurMat=new THREE.ShaderMaterial({uniforms:{tInput:{value:null},direction:{value:new THREE.Vector2(1,0)},texel:{value:new THREE.Vector2(1,1)}},vertexShader:`varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}`,fragmentShader:`precision highp float;uniform sampler2D tInput;uniform vec2 direction;uniform vec2 texel;varying vec2 vUv;void main(){vec2 d=direction*texel;vec3 c=texture2D(tInput,vUv).rgb*.227027;c+=texture2D(tInput,vUv+d*1.384615).rgb*.316216;c+=texture2D(tInput,vUv-d*1.384615).rgb*.316216;c+=texture2D(tInput,vUv+d*3.230769).rgb*.070270;c+=texture2D(tInput,vUv-d*3.230769).rgb*.070270;gl_FragColor=vec4(c,1.0);}`});
const compositeMat=new THREE.ShaderMaterial({uniforms:{tBase:{value:null},tBodyMask:{value:null},tBloom:{value:null},tBloomWide:{value:null},bloom:{value:FIXED_BLOOM},exposure:{value:FIXED_EXPOSURE},resolution:{value:new THREE.Vector2(1,1)}},vertexShader:`varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}`,fragmentShader:`precision highp float;uniform sampler2D tBase;uniform sampler2D tBodyMask;uniform sampler2D tBloom;uniform sampler2D tBloomWide;uniform float bloom;uniform float exposure;uniform vec2 resolution;varying vec2 vUv;vec3 aces(vec3 x){const float a=2.51,b=.03,c=2.43,d=.59,e=.14;return clamp((x*(a*x+b))/(x*(c*x+d)+e),0.,1.);}void main(){vec3 base=texture2D(tBase,vUv).rgb;float bodyMask=smoothstep(0.02,0.98,texture2D(tBodyMask,vUv).r);vec3 bloomField=texture2D(tBloom,vUv).rgb+texture2D(tBloomWide,vUv).rgb*.72;float bodyEmission=smoothstep(1.15,3.5,max(max(base.r,base.g),base.b));float bloomGate=max(1.0-bodyMask,bodyEmission);vec3 c=base+bloomField*bloom*bloomGate;c=aces(c*exposure);float vig=1.0-smoothstep(.25,.78,length(vUv-.5));c*=.86+.14*vig;c=pow(c,vec3(1.0/2.2));gl_FragColor=vec4(c,1.0);}`});
function makeRT(w,h,samples=0){
  const rt=new THREE.WebGLRenderTarget(Math.max(1,w),Math.max(1,h),{
    type:renderTargetType,format:THREE.RGBAFormat,minFilter:THREE.LinearFilter,magFilter:THREE.LinearFilter,depthBuffer:true,
  });
  rt.samples=isWebGL2?Math.min(Math.max(0,samples),maxSamples):0;return rt;
}
function disposeTargets(){for(const rt of [sceneRT,bodyMaskRT,brightRT,blurA,blurB,smallA,smallB])if(rt)rt.dispose();}
function resize(){
  const cssW=Math.max(2,root.clientWidth),cssH=Math.max(2,root.clientHeight);renderer.setSize(cssW,cssH,false);camera.aspect=cssW/cssH;camera.updateProjectionMatrix();
  const drawSize=new THREE.Vector2();renderer.getDrawingBufferSize(drawSize);const w=Math.max(2,Math.floor(drawSize.x)),h=Math.max(2,Math.floor(drawSize.y));
  disposeTargets();
  try{
    sceneRT=makeRT(w,h,4);bodyMaskRT=makeRT(w,h,4);brightRT=makeRT(Math.floor(w/2),Math.floor(h/2));blurA=makeRT(Math.floor(w/2),Math.floor(h/2));blurB=makeRT(Math.floor(w/2),Math.floor(h/2));smallA=makeRT(Math.floor(w/4),Math.floor(h/4));smallB=makeRT(Math.floor(w/4),Math.floor(h/4));
    compositeMat.uniforms.resolution.value.set(w,h);postprocessingEnabled=true;
  }catch(error){postprocessingEnabled=false;setStatus('Basic rendering active — this GPU does not support the requested HDR target configuration.');}
}
window.addEventListener('resize',resize);
function pass(material,target){postQuad.material=material;renderer.setRenderTarget(target);renderer.clear();renderer.render(postScene,postCamera);}
function renderBasic(){renderer.setRenderTarget(null);renderer.clear();renderer.render(scene,camera);}
function renderPost(){
  if(!postprocessingEnabled){renderBasic();return;}
  try{
    renderer.setRenderTarget(sceneRT);renderer.clear();renderer.render(scene,camera);
    renderer.setRenderTarget(bodyMaskRT);renderer.clear();renderer.render(bodyMaskScene,camera);
    brightMat.uniforms.tInput.value=sceneRT.texture;pass(brightMat,brightRT);
    blurMat.uniforms.texel.value.set(1/brightRT.width,1/brightRT.height);blurMat.uniforms.tInput.value=brightRT.texture;blurMat.uniforms.direction.value.set(1,0);pass(blurMat,blurA);blurMat.uniforms.tInput.value=blurA.texture;blurMat.uniforms.direction.value.set(0,1);pass(blurMat,blurB);
    brightMat.uniforms.tInput.value=blurB.texture;pass(brightMat,smallA);blurMat.uniforms.texel.value.set(1/smallA.width,1/smallA.height);blurMat.uniforms.tInput.value=smallA.texture;blurMat.uniforms.direction.value.set(1,0);pass(blurMat,smallB);blurMat.uniforms.tInput.value=smallB.texture;blurMat.uniforms.direction.value.set(0,1);pass(blurMat,smallA);
    compositeMat.uniforms.tBase.value=sceneRT.texture;compositeMat.uniforms.tBodyMask.value=bodyMaskRT.texture;compositeMat.uniforms.tBloom.value=blurB.texture;compositeMat.uniforms.tBloomWide.value=smallA.texture;postQuad.material=compositeMat;renderer.setRenderTarget(null);renderer.clear();renderer.render(postScene,postCamera);
  }catch(error){postprocessingEnabled=false;setStatus('HDR postprocessing was disabled after a GPU capability error; physical geometry and shading remain active.');renderBasic();}
}

resize();applyPhotometryMode();applyElapsed(0);preset(CONFIG.mode==='solar'&&CONFIG.scope==='local'?'observer':(CONFIG.mode==='solar'?'earth':'moon'));setGroupVisibility();
const capabilityText=`${isWebGL2?'WebGL2':'WebGL1'} · ${hasHalfFloatColor?'HDR half-float':'8-bit fallback'} · ${Math.min(4,maxSamples||0)}x MSAA target`;
setStatus(`Ready — ${capabilityText}; UTC playback is uniform, LUT-accelerated limb-darkened surface photometry is active, and the camera follows its physical target${rendererFallback?' (fallback renderer)':''}`);
let last=performance.now();
function animate(now){
  const dt=Math.min(.1,(now-last)/1000);last=now;sunMaterial.uniforms.uTime.value=now/1000;
  localBeams.halo.material.opacity=.12*(.92+.08*Math.sin(now*.0017));fullBeams.halo.material.opacity=.12*(.92+.08*Math.sin(now*.0017));
  if(playing&&CONFIG.states.length>1&&eventDuration>0){
    const playbackDuration=Math.max(.1,Number(CONFIG.playback_seconds||30));
    const rate=eventDuration/playbackDuration;
    elapsedSeconds+=dt*rate;
    if(elapsedSeconds>eventDuration)elapsedSeconds=0;
    applyElapsed(elapsedSeconds);
  }
  updateLabelScales();renderPost();requestAnimationFrame(animate);
}
requestAnimationFrame(animate);

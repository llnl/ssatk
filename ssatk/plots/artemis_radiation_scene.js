/* Mission overview and 3D demo; trajectory by SSAPy, dose only by IRENE. */
const $ = id => document.getElementById(id);
const viewport = $('viewport'), canvas = $('scene');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:false});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x050b14);
THREE.ColorManagement.enabled = false;
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 2000);
camera.up.set(0,0,1);
scene.add(new THREE.AmbientLight(0x8aaaca,0.65*Math.PI));
const sun = new THREE.DirectionalLight(0xffeedb,1.5*Math.PI);
sun.position.set(-12,-22,28);scene.add(sun);
const world = new THREE.Group();scene.add(world);
const lastMoon = new THREE.Vector3(...DATA.moon[DATA.moon.length-1]);
const xhat = lastMoon.clone().normalize();
const zhat = new THREE.Vector3(...DATA.r[0]).cross(xhat).normalize();
const yhat = zhat.clone().cross(xhat).normalize();
const basis = new THREE.Matrix4().makeBasis(xhat,yhat,zhat);
world.quaternion.setFromRotationMatrix(basis).invert();
const starGeometry = new THREE.BufferGeometry();
starGeometry.setAttribute('position',new THREE.Float32BufferAttribute(DATA.stars.v.map(value=>value*800),3));
starGeometry.setAttribute('color',new THREE.Float32BufferAttribute(DATA.stars.c,3));
starGeometry.setAttribute('starSize',new THREE.Float32BufferAttribute(DATA.stars.s,1));
const stars = new THREE.Points(starGeometry,new THREE.ShaderMaterial({
  uniforms:{pixelRatio:{value:renderer.getPixelRatio()}},
  // Enlarge catalog markers so they survive movie compression and reduced playback size.
  vertexShader:`uniform float pixelRatio; attribute float starSize; varying vec3 vColor;
    void main(){vColor=color; gl_PointSize=starSize*2.0*pixelRatio;
      gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
  fragmentShader:`varying vec3 vColor;
    void main(){float alpha=1.0-smoothstep(0.12,0.5,length(gl_PointCoord-vec2(0.5)));
      if(alpha<=0.01)discard; gl_FragColor=vec4(vColor,alpha);
      #include <colorspace_fragment>
    }`,
  vertexColors:true,transparent:true,depthTest:true,depthWrite:false
}));
stars.quaternion.copy(world.quaternion);stars.frustumCulled=false;stars.renderOrder=-1000;
scene.add(stars);
const earthGroup = new THREE.Group();world.add(earthGroup);
const matrix = DATA.rotation;
const earthRotation = new THREE.Matrix4().set(
  ...matrix[0],0,...matrix[1],0,...matrix[2],0,0,0,0,1);
const earthStart = new THREE.Quaternion().setFromRotationMatrix(earthRotation);
const sphere = new THREE.SphereGeometry(1,96,64);sphere.rotateX(Math.PI/2);
const earth = new THREE.Mesh(sphere,new THREE.MeshPhongMaterial({color:0xffffff,shininess:12}));
earth.scale.set(1,1,0.996647);earthGroup.add(earth);
const halo = new THREE.Mesh(new THREE.SphereGeometry(1.025,64,32),
  new THREE.MeshBasicMaterial({color:0x559bcc,transparent:true,opacity:0.08,side:THREE.BackSide}));
earthGroup.add(halo);
for(const belt of DATA.belts){
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(belt.positions.flat(),3));
  geometry.setIndex(belt.indices);geometry.computeVertexNormals();
  const color = new THREE.Color(belt.color);
  earthGroup.add(new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color,
    transparent:true,opacity:0.16,side:THREE.DoubleSide,depthWrite:false})));
  earthGroup.add(new THREE.LineSegments(new THREE.WireframeGeometry(geometry),
    new THREE.LineBasicMaterial({color,transparent:true,opacity:0.035,depthWrite:false})));
}
const moon = new THREE.Mesh(new THREE.SphereGeometry(1737.4/DATA.earth_radius_km,48,32),
  new THREE.MeshPhongMaterial({color:0xc5c9d2,shininess:2}));world.add(moon);
const pathGeometry = new THREE.BufferGeometry().setFromPoints(DATA.r.map(row=>new THREE.Vector3(...row)));
world.add(new THREE.Line(pathGeometry,new THREE.LineBasicMaterial({color:0x4c8b98,transparent:true,opacity:0.32})));
const trailGeometry = pathGeometry.clone();
const trailPositions = trailGeometry.attributes.position;
let trailTipIndex = null;
world.add(new THREE.Line(trailGeometry,new THREE.LineBasicMaterial({color:0x82f6ec})));
const craft = new THREE.Mesh(new THREE.SphereGeometry(0.035,16,12),new THREE.MeshBasicMaterial({color:0xeaffff}));
world.add(craft);
const ring = new THREE.Mesh(new THREE.RingGeometry(0.075,0.083,48),
  new THREE.MeshBasicMaterial({color:0x86ffff,transparent:true,opacity:0.8,side:THREE.DoubleSide}));
world.add(ring);
const chart = $('chart'), ctx = chart.getContext('2d');
const pending = !DATA.dose;
$('notice').textContent = DATA.belts.length ? 'IRENE mean flux contours at departure, rotating with Earth. Hypothetical outbound trajectory.' : 'Radiation belts pending official IRENE. Hypothetical outbound trajectory.';
$('status').textContent = pending ? 'IRENE input files prepared. Official runtime required. Solar particle events and galactic cosmic rays are outside this calculation.' :
  `IRENE V${DATA.dose.metadata.version} · AE9 + AP9 mean · SHIELDOSE2. Illustrative tissue dose; SEP/GCR excluded. The supplied IGRF field is fixed after 2020.`;
if(!pending){$('dose').classList.remove('pending');$('chartnote').classList.remove('pending');}
if(DATA.belts.length){
  $('status').textContent += ' Contours: '+DATA.belts.map(belt=>`${belt.metadata.model} ${belt.metadata.contour_flux.toPrecision(3)} cm⁻² s⁻¹`).join('; ')+'.';
}
let movieTime = 0, playing = false, previous = performance.now();
const clamp = (value,low,high)=>Math.min(high,Math.max(low,value));
const smooth = value=>{value=clamp(value,0,1);return value*value*(3-2*value);};
function sample(rows,index,fraction){return new THREE.Vector3(...rows[index]).lerp(new THREE.Vector3(...rows[Math.min(index+1,rows.length-1)]),fraction);}
function drawChart(day){
  const width = chart.clientWidth, height = chart.clientHeight;
  chart.width=width*2;chart.height=height*2;ctx.scale(2,2);ctx.clearRect(0,0,width,height);
  const left=38,top=14,right=width-8,bottom=height-29;
  ctx.strokeStyle='#34475e';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,bottom);ctx.lineTo(right,bottom);ctx.stroke();
  ctx.font='10px system-ui';ctx.fillStyle='#8ba0bb';ctx.fillText('mGy',5,12);ctx.fillText('Mission day',width/2-18,height-2);
  for(let tick=0;tick<=4;tick++){const x=left+(right-left)*tick/4;ctx.fillText(String(tick),x-3,bottom+15);}
  if(pending){
    ctx.fillStyle='#b8a68c';ctx.font='12px system-ui';ctx.textAlign='center';
    ctx.fillText('Awaiting IRENE output',(left+right)/2,(top+bottom)/2);ctx.textAlign='left';
    ctx.fillStyle='#8ba0bb';ctx.fillText('—',14,bottom);return;
  }
  const values=DATA.dose.total_mgy.map(row=>row[1]);
  const maximum=Math.max(...values,1e-12)*1.1;
  ctx.fillText('0',21,bottom);ctx.fillText(maximum.toPrecision(2),1,top+7);
  ctx.strokeStyle='#6fdfeb';ctx.lineWidth=2;ctx.beginPath();
  DATA.dose.mission_day.forEach((t,index)=>{if(t<=day){const x=left+(right-left)*t/4,y=bottom-(bottom-top)*values[index]/maximum;if(index===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}});ctx.stroke();
  ctx.strokeStyle='#6fdfeb55';ctx.beginPath();const cursor=left+(right-left)*day/4;ctx.moveTo(cursor,top);ctx.lineTo(cursor,bottom);ctx.stroke();
}
function placeLabel(id,position,offset){
  const projected=position.clone().applyMatrix4(world.matrixWorld).project(camera);
  const element=$(id);element.style.display=(Math.abs(projected.x)<1&&Math.abs(projected.y)<1&&projected.z<1)?'block':'none';
  element.style.left=`${(projected.x+1)*viewport.clientWidth/2+offset}px`;
  element.style.top=`${(1-projected.y)*viewport.clientHeight/2+16}px`;
}
window.setMovieTime=function(value){
  movieTime=clamp(value,0,42);
  const overview=movieTime<10;
  $('intro').style.display=overview?'grid':'none';
  $('intro').inert=!overview;
  document.body.classList.toggle('overview',overview);
  document.querySelector('main').inert=overview;
  document.querySelector('footer').inert=overview;
  const seconds=movieTime<10?0:movieTime<20?(movieTime-10)/10*21600:21600+clamp((movieTime-20)/19,0,1)*(DATA.seconds.at(-1)-21600);
  let index=0;while(index<DATA.seconds.length-2&&DATA.seconds[index+1]<seconds)index++;
  const fraction=clamp((seconds-DATA.seconds[index])/(DATA.seconds[index+1]-DATA.seconds[index]),0,1);
  const position=sample(DATA.r,index,fraction);craft.position.copy(position);ring.position.copy(position);
  // Restore the previous partial segment before moving or scrubbing the tip.
  if(trailTipIndex!==null)trailPositions.setXYZ(trailTipIndex,...DATA.r[trailTipIndex]);
  trailTipIndex=index+1;
  trailPositions.setXYZ(trailTipIndex,position.x,position.y,position.z);trailPositions.needsUpdate=true;
  moon.position.copy(sample(DATA.moon,index,fraction));
  trailGeometry.setDrawRange(0,index+2);
  const spin=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1),DATA.earth_omega*seconds);
  earthGroup.quaternion.copy(earthStart).multiply(spin);
  const pullback=smooth((movieTime-14)/12), distance=12+(lastMoon.length()*1.18-12)*pullback;
  const focus=new THREE.Vector3(lastMoon.length()*0.47*pullback,0,0);
  camera.position.copy(focus).add(new THREE.Vector3(-distance*0.05,-distance,distance*0.52));camera.lookAt(focus);
  // Center the distant sky on the camera so the lunar transfer introduces no stellar parallax.
  stars.position.copy(camera.position);
  const markerScale=1+pullback*4;craft.scale.setScalar(markerScale);ring.scale.setScalar(markerScale);
  ring.quaternion.copy(world.quaternion).invert().multiply(camera.quaternion);
  $('day').textContent=(seconds/86400).toFixed(2);
  $('distance').textContent=Math.round(position.length()*DATA.earth_radius_km).toLocaleString()+' km';
  $('phase').textContent=seconds<1?'Translunar injection':position.length()<6?'Crossing the radiation belt region':seconds<DATA.seconds.at(-1)-1?'Translunar coast':'Lunar approach';
  if(!pending){
    const d=DATA.dose.total_mgy, dose=d[index][1]*(1-fraction)+d[Math.min(index+1,d.length-1)][1]*fraction;
    $('dose').textContent=dose.toPrecision(3)+' mGy';$('chartnote').textContent='AE9 + AP9 cumulative mean · 3 mm Al';
  }
  drawChart(seconds/86400);
  $('scrub').value=movieTime;$('clock').textContent=`00:${String(Math.floor(movieTime)).padStart(2,'0')} / 00:42`;
  scene.updateMatrixWorld(true);camera.updateMatrixWorld();
  placeLabel('earthlabel',new THREE.Vector3(),18);placeLabel('moonlabel',moon.position,16);
  renderer.render(scene,camera);
};
function resize(){renderer.setSize(viewport.clientWidth,viewport.clientHeight,false);camera.aspect=viewport.clientWidth/viewport.clientHeight;camera.updateProjectionMatrix();window.setMovieTime(movieTime);}
window.addEventListener('resize',resize);
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause':'Play';};
$('start').onclick=()=>{window.setMovieTime(10);playing=true;previous=performance.now();$('play').textContent='Pause';$('scene').focus();};
$('restart').onclick=()=>{playing=false;window.setMovieTime(0);$('play').textContent='Play';$('start').focus();};
$('scrub').oninput=event=>{playing=false;$('play').textContent='Play';window.setMovieTime(Number(event.target.value));};
function animate(now){const delta=(now-previous)/1000;previous=now;if(playing&&!window.demoRecording){window.setMovieTime(movieTime+Math.min(delta,0.1));if(movieTime>=42){playing=false;$('play').textContent='Play';}}requestAnimationFrame(animate);}
resize();
new THREE.TextureLoader().load(DATA.earth_texture,texture=>{
  texture.colorSpace=THREE.SRGBColorSpace;texture.anisotropy=renderer.capabilities.getMaxAnisotropy();
  earth.material.map=texture;earth.material.needsUpdate=true;window.demoReady=true;$('start').disabled=false;
  $('intro-status').textContent=pending?'Scene ready. Radiation analysis pending official IRENE.':'Scene ready. IRENE radiation analysis available.';
  window.setMovieTime(movieTime);requestAnimationFrame(animate);
},undefined,error=>{$('intro-status').textContent='Earth texture failed to load. Reload to try again.';$('notice').textContent='Earth texture failed to load';console.error(error);});

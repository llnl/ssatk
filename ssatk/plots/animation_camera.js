/* Shared Three.js animation shots. Inputs use the viewer's Cartesian scene
 * coordinates and kilometres. Camera changes never touch simulation state. */
class SSATKAnimationCamera {
  constructor(camera, config, up, radius) {
    this.camera = camera;
    this.config = config;
    this.mode = config.mode;
    this.up = new THREE.Vector3(...(config.up || up)).normalize();
    this.center = new THREE.Vector3();
    this.radius = radius;
    this.savedEye = camera.position.clone();
    this.lastTangent = null;
    const el = THREE.MathUtils.degToRad(config.elevation);
    const az = THREE.MathUtils.degToRad(config.azimuth);
    // The generated default up is +Y for the Earth viewer, +Z for Moon.
    this.direction = new THREE.Vector3(Math.cos(el)*Math.cos(az), Math.cos(el)*Math.sin(az), Math.sin(el));
    if (Math.abs(this.up.y) > .999) this.direction.set(this.direction.x, this.direction.z, -this.direction.y);
    this.initialEye = null;
    if (config.mode === 'overview' && (config.offset || config.distance)) {
      this.initialEye = config.offset ? new THREE.Vector3(...config.offset) : this.direction.clone().multiplyScalar(config.distance);
      camera.position.copy(this.initialEye); camera.up.copy(this.up); camera.lookAt(this.center);
      this.savedEye.copy(this.initialEye);
    }
    this.panel = document.createElement('div');
    this.panel.style.cssText = 'position:fixed;right:14px;top:14px;z-index:10000;padding:9px 12px;background:#0b1424ee;color:#eee;border:1px solid #536782;border-radius:6px;font:13px sans-serif';
    const label = document.createElement('label'); label.textContent = 'Camera ';
    this.select = document.createElement('select'); this.select.id = 'camera-shot';
    for (const [value, text] of [['overview','Overview'],['follow','Fixed follow'],['chase','Chase'],['orbit','Orbit sweep']]) {
      const option = document.createElement('option'); option.value = value; option.textContent = text; this.select.appendChild(option);
    }
    this.select.value = this.mode;
    label.appendChild(this.select); this.panel.appendChild(label);
    this.status = document.createElement('div'); this.status.style.cssText = 'font-size:11px;color:#b6c7dc;margin-top:4px';
    this.targetSelect = document.createElement('select'); this.targetSelect.id = 'camera-target';
    this.targetSelect.title = 'Object to follow'; this.targetSelect.style.cssText = 'display:block;margin-top:6px;max-width:200px';
    this.panel.appendChild(this.targetSelect);
    this.targetSelect.onchange = () => { this.subject = this.subjects[Number(this.targetSelect.value)]; this.lastTangent = null; };
    this.panel.appendChild(this.status);
    const container = document.getElementById('sat-panel') || document.getElementById('orbit-toolbar-body');
    if (container) {
      this.panel.style.cssText = 'padding:10px 0;margin-bottom:10px;color:#eee;border-bottom:1px solid #536782;font:13px sans-serif';
      container.prepend(this.panel);
    } else document.body.appendChild(this.panel);
    this.select.onchange = () => {
      this.mode = this.select.value; this.lastTangent = null;
      if (this.onModeChange) this.onModeChange(this.mode);
      if (this.mode === 'overview') {
        this.camera.position.copy(this.savedEye); this.camera.up.copy(this.up); this.camera.lookAt(this.center);
      }
    };
  }
  selectTarget(subjects, label) {
    const labels = subjects.map(label);
    // Names need not be unique. A replaced/reordered object must refresh the
    // selection even when its label is identical to the previous object's.
    const changed = !this.subjects || subjects.length !== this.subjects.length ||
      subjects.some((subject, i) => subject !== this.subjects[i] || labels[i] !== this.subjectLabels[i]);
    if (changed) {
      const previous = this.subject;
      if (!this.subject || !subjects.includes(this.subject)) this.subject = subjects[this.config.target];
      if (this.subject !== previous) this.lastTangent = null;
      this.subjects = subjects.slice(); this.subjectLabels = labels;
      this.targetSelect.replaceChildren();
      subjects.forEach((subject, index) => {
        const option = document.createElement('option'); option.value = index; option.textContent = labels[index]; this.targetSelect.appendChild(option);
      });
      this.targetSelect.value = String(subjects.indexOf(this.subject));
    }
    this.targetSelect.style.display = ['follow','chase'].includes(this.mode) ? 'block' : 'none';
    return this.subject;
  }
  update(target, tangent, phase, radius, name='target') {
    this.radius = radius;
    if (this.mode === 'overview') { this.status.textContent = 'Full scene · drag to adjust'; return; }
    if ((this.mode === 'follow' || this.mode === 'chase') && !target) {
      this.status.textContent = 'Select a visible moving object'; return;
    }
    const center = this.mode === 'orbit' ? this.center : target;
    const verticalHalfFov = THREE.MathUtils.degToRad(this.camera.fov/2);
    const horizontalHalfFov = Math.atan(Math.tan(verticalHalfFov)*this.camera.aspect);
    const fitDistance = 1.15*radius/Math.sin(Math.min(verticalHalfFov,horizontalHalfFov));
    const distance = this.config.distance || (this.mode === 'orbit'
      ? Math.max(this.savedEye.length(),fitDistance) : Math.max(radius*.35,1));
    let offset = this.config.offset ? new THREE.Vector3(...this.config.offset) : this.direction.clone().multiplyScalar(distance);
    if (this.mode === 'orbit') offset.applyAxisAngle(this.up, 2*Math.PI*this.config.turns*phase);
    if (this.mode === 'chase') {
      if (tangent && tangent.lengthSq() > 1e-24) this.lastTangent = tangent.clone().normalize();
      if (!this.lastTangent) { this.status.textContent = 'Chase needs a moving target'; return; }
      offset = this.lastTangent.clone().negate().addScaledVector(this.up,.3).normalize().multiplyScalar(distance);
    }
    this.camera.position.copy(center).add(offset);
    this.camera.up.copy(this.up);
    const direction = offset.clone().normalize();
    if (Math.abs(direction.dot(this.up)) > 1-1e-10) {
      const axes = [new THREE.Vector3(1,0,0),new THREE.Vector3(0,1,0),new THREE.Vector3(0,0,1)];
      axes.sort((a,b)=>Math.abs(a.dot(direction))-Math.abs(b.dot(direction))); this.camera.up.copy(axes[0]);
    }
    this.camera.lookAt(center);
    this.status.textContent = this.mode === 'orbit' ? 'Full scene · orbit sweep' : name;
  }
}

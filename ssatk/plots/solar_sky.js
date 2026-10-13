(function () {
    "use strict";
    const gd = document.getElementById('{plot_id}');
    const sky = gd.layout.meta.solar_sky;
    let camera = gd.layout.scene.camera;
    let scheduled = false;
    let previous = '';
    const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
    const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
    const unit = v => { const n = Math.hypot(...v); return v.map(x => x/n); };
    const vec = (obj, fallback) => ['x','y','z'].map((k, i) => obj?.[k] ?? fallback[i]);
    function draw() {
        scheduled = false;
        const margin = gd.layout.margin || {};
        const w = Math.max(1, gd._fullLayout?._size?.w ?? (gd.clientWidth - (margin.l || 0) - (margin.r || 0)));
        const h = Math.max(1, gd._fullLayout?._size?.h ?? (gd.clientHeight - (margin.t || 0) - (margin.b || 0)));
        const signature = JSON.stringify([camera, w, h]);
        if (signature === previous) return;
        previous = signature;
        const eye = vec(camera.eye, [1.25,1.25,1.25]);
        const center = vec(camera.center, [0,0,0]);
        const forward = unit(center.map((x, i) => x-eye[i]));
        let right = cross(forward, vec(camera.up, [0,0,1]));
        if (Math.hypot(...right) < 1e-12) right = cross(forward, [0,1,0]);
        right = unit(right);
        const up = cross(right, forward);
        const focal = h / (2 * Math.tan(Math.PI / 8));
        const canvas = document.createElement('canvas');
        canvas.width = w; canvas.height = h;
        const ctx = canvas.getContext('2d');
        sky.directions.forEach((v, i) => {
            const z = dot(v, forward);
            if (z <= 0) return;
            const x = w/2 + focal*dot(v, right)/z;
            const y = h/2 - focal*dot(v, up)/z;
            if (x < 0 || x > w || y < 0 || y > h) return;
            ctx.fillStyle = sky.colors[i];
            ctx.beginPath(); ctx.arc(x, y, sky.sizes[i], 0, 2*Math.PI); ctx.fill();
        });
        Plotly.relayout(gd, {'images[0].source': canvas.toDataURL('image/png')});
    }
    function schedule(event) {
        camera = event?.['scene.camera'] || event?.frame?.layout?.scene?.camera || gd._fullLayout?.scene?.camera || gd.layout.scene.camera;
        if (!scheduled) { scheduled = true; requestAnimationFrame(draw); }
    }
    gd.on('plotly_relayout', schedule);
    gd.on('plotly_relayouting', schedule);
    gd.on('plotly_animatingframe', schedule);
    gd.on('plotly_animated', schedule);
    gd.on('plotly_afterplot', schedule);
    const observer = new ResizeObserver(() => schedule());
    observer.observe(gd);
    schedule();
}());

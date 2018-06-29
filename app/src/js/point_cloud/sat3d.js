/* global Sat THREE SatPointCloud Box3d */

/**
 * Point Cloud 3DBBox Sat
 */
function Sat3d() {
    this.slider = document.getElementById('pc_scroll');
    this.container = document.getElementById('main_container');

    this.renderer = new THREE.WebGLRenderer();
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(this.container.offsetWidth,
        this.container.offsetHeight);
    this.container.appendChild(this.renderer.domElement);

    this.view_params = [
        {
            top: 0,
            left: 0,
            width: 0.6,
            height: 1.0,
            restrictDrag: false,
            position: new THREE.Vector3(0, -8, 0.5),
        },
        {
            top: 0,
            left: 0.6,
            width: 0.4,
            height: 0.5,
            restrictDrag: true,
            position: new THREE.Vector3(0, 0, 4),
        },
        {
            top: 0.5,
            left: 0.6,
            width: 0.4,
            height: 0.5,
            restrictDrag: true,
            position: new THREE.Vector3(0, -8, 0),
        },
    ];

    Sat.call(this, SatPointCloud, Box3d);
}

Sat3d.prototype = Object.create(Sat.prototype);

Sat3d.prototype.newItem = function(url) {
    let item = new this.ItemType(this, this.items.length, url);
    this.items.push(item);
    return item;
};

Sat3d.prototype.loaded = function() {
    Sat.prototype.loaded.call(this);
    this.slider.min = 1;
    this.slider.max = this.items.length;
    this.slider.value = 1;

    this.animate();
};

Sat3d.prototype.animate = function() {
    requestAnimationFrame(this.animate.bind(this));
    this.currentItem.redraw();
};

/* global THREE */

import {Sat} from '../sat';
import {SatPointCloud} from './sat_point_cloud';
import {Box3d} from './box3d';


/**
 *
 * @param {SatItem} itemType
 * @param {SatLabel} labelType
 * @constructor
 */
export function Sat3d() {
    this.slider = document.getElementById('slider');
    this.container = document.getElementById('main_container');

    this.renderer = new THREE.WebGLRenderer();
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(this.container.offsetWidth,
        this.container.offsetHeight);
    this.container.appendChild(this.renderer.domElement);

    window.addEventListener( 'resize', this.onWindowResize.bind(this), false );

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

Sat3d.prototype.onWindowResize = function() {
    for (let i=0; i<this.items.length; i++) {
        for (let j=0; j<this.items[i].views.length; j++) {
            let camera = this.items[i].views[j].camera;
            camera.aspect = this.container.offsetWidth /
                this.container.offsetHeight;
            camera.updateProjectionMatrix();
        }
    }

    this.renderer.setSize(this.container.offsetWidth,
        this.container.offsetHeight);
};

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

    // Load point cloud data
    this.items[0].getPCJSON();

    this.animate();
};

Sat3d.prototype.animate = function() {
    requestAnimationFrame(this.animate.bind(this));
    this.currentItem.redraw();
};


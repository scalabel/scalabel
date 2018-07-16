/* global SatLabel THREE */

/**
 * Label for Point Cloud 3D BBox
 * @param {Sat} sat: context
 * @param {int} id: label id
 */
function Box3d(sat, id) {
    SatLabel.call(this, sat, id);
    this.categoryPath = '';
    this.categoryArr = [];
}

Box3d.prototype = Object.create(SatLabel.prototype);

Box3d.prototype.setColor = function(hexColor, faces=null) {
    if (faces != null) {
        for (let i = 0; i < faces.length; i++) {
            this.box.geometry.faces[faces[i]].color.setHex(hexColor);
        }
    } else {
        for (let i = 0; i < this.box.geometry.faces.length; i++) {
            this.box.geometry.faces[i].color.setHex(hexColor);
        }
    }
    this.box.geometry.colorsNeedUpdate = true;
};

Box3d.prototype.moveBoxAlongViewPlane = function(viewPlaneNormal,
                                                           viewPlaneOffset,
                                                           boxMouseOverPoint,
                                                           cameraPosition,
                                                           projection) {
    // Get vector from camera to box
    let camToBox = new THREE.Vector3();
    camToBox.copy(cameraPosition);
    camToBox.sub(boxMouseOverPoint);

    // Adjust projection to go from camera to a plane parallel to the
    // screen that intersects the box's position
    let dist = -camToBox.dot(viewPlaneNormal) /
        projection.dot(viewPlaneNormal);
    projection.multiplyScalar(dist);

    // Set box position to point
    projection.add(cameraPosition);
    projection.add(viewPlaneOffset);

    this.box.position.copy(projection);
    this.box.outline.position.copy(projection);
};

Box3d.prototype.scaleBox = function(cameraPosition, projection) {
    let worldToModel = new THREE.Matrix4();
    worldToModel.getInverse(this.box.matrixWorld);

    // Get camera position in model space
    let cameraPosModel = new THREE.Vector3();
    cameraPosModel.copy(cameraPosition);
    cameraPosModel.applyMatrix4(worldToModel);

    // Get projection from camera to world
    worldToModel.setPosition(new THREE.Vector3(0, 0, 0));
    projection.applyMatrix4(worldToModel);

    // Adjust projection to go from camera position to either the top
    // or the bottom of the box, depending on which is closer
    let distToTop = -cameraPosModel.z;
    let dist = distToTop / projection.z;
    projection.multiplyScalar(dist);

    // Get point on aforementioned plane
    let boxPlanePoint = new THREE.Vector3();
    boxPlanePoint.copy(cameraPosModel);
    boxPlanePoint.add(projection);

    // Set closest box corner to be at the point
    this.box.scale.x = boxPlanePoint.x * this.box.scale.x * 2;
    this.box.outline.scale.x = boxPlanePoint.x * this.box.scale.x * 2;
    this.box.scale.y = boxPlanePoint.y * this.box.scale.y * 2;
    this.box.outline.scale.y = boxPlanePoint.y * this.box.scale.y * 2;
};

Box3d.prototype.extrudeBox = function(boxMouseOverPoint,
                                                cameraPosition, projection) {
    // Get vertical distance from B to current mouse raycast
    let vertDist = cameraPosition.z - boxMouseOverPoint.z +
        (boxMouseOverPoint.dot(projection) - projection.dot(cameraPosition)) *
        projection.z;
    vertDist /= 1 - projection.z * projection.z;

    // Scale
    let scale = (vertDist + boxMouseOverPoint.z - this.box.position.z) * 2;
    this.box.scale.z = scale;
    this.box.outline.scale.z = scale;
};

// Rotate box on z axis
Box3d.prototype.rotateBox = function(cameraPosition,
                                               cameraWorldDirection,
                                               currentProjection,
                                               newProjection) {
    // Remember that (mouseX, mouseY) must intersect with box
    let newRay = new THREE.Ray(cameraPosition, newProjection);

    let currentRay = new THREE.Ray(cameraPosition, currentProjection);

    let norm = cameraWorldDirection;
    norm.normalize();
    let viewPlane = new THREE.Plane(norm, norm.dot(this.box.position));
    let currentHit = currentRay.intersectPlane(viewPlane);
    let newHit = newRay.intersectPlane(viewPlane);
    currentHit.sub(this.box.position);
    newHit.sub(this.box.position);

    if (currentHit != null && newHit != null) {
        let angle = currentHit.angleTo(newHit);
        let posCross = new THREE.Vector3();
        posCross.copy(currentHit);
        posCross.cross(newHit);
        if (posCross.dot(norm) < 0) {
            angle = angle * (-1);
        }
        let q = new THREE.Quaternion();
        q.setFromAxisAngle(norm, angle);
        this.box.quaternion.premultiply(q);
        this.box.outline.quaternion.premultiply(q);
        this.box.rotation.setFromQuaternion(this.box.quaternion);
        this.box.outline.rotation.setFromQuaternion(
            this.box.outline.quaternion);
    }
};

Box3d.prototype.toJson = function() {
    let label = this.encodeBaseJson();

    this.data = {};

    this.data.position = [this.box.position.x, this.box.position.y,
                           this.box.position.z];

    this.data.rotation = [this.box.rotation.x, this.box.rotation.y,
                           this.box.rotation.z];

    this.data.scale = [this.box.scale.x, this.box.scale.y,
                        this.box.scale.z];

    label.data = this.data;
    label.categoryPath = this.categoryPath;

    return label;
};

Box3d.prototype.fromJsonVariables = function(json) {
    this.decodeBaseJsonVariables(json);
    this.data = json.data;
    this.categoryPath = json.categoryPath;
    this.categoryArr = this.categoryPath.split(',');
    this.name = this.categoryArr[this.categoryArr.length - 1];
};

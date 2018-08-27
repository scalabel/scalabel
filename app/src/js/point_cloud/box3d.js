/* global THREE */

import {SatLabel} from '../sat';

/**
 * Label for Point Cloud 3D BBox
 * @param {Sat} sat: context
 * @param {int} id: label id
 */
export function Box3d(sat, id) {
  SatLabel.call(this, sat, id);
  this.categoryPath = '';
  this.categoryArr = [];
}

Box3d.prototype = Object.create(SatLabel.prototype);

Box3d.prototype.color = function() {
  let color = SatLabel.prototype.color.call(this);
  let newColor = [];
  for (let i = 0; i < color.length; i++) {
    newColor.push(color[i] / 255.0);
  }
  return newColor;
};

Box3d.prototype.setColor = function(hexColor, faces = null) {
  let inputArray = typeof hexColor == 'object';
  if (faces != null) {
    for (let i = 0; i < faces.length; i++) {
      if (inputArray) {
        this.box.geometry.faces[faces[i]].color.fromArray(hexColor);
      } else {
        this.box.geometry.faces[faces[i]].color.set(hexColor);
      }
    }
  } else {
    for (let i = 0; i < this.box.geometry.faces.length; i++) {
      if (inputArray) {
        this.box.geometry.faces[i].color.fromArray(hexColor);
      } else {
        this.box.geometry.faces[i].color.set(hexColor);
      }
    }
  }
  this.box.geometry.colorsNeedUpdate = true;
};

Box3d.prototype.createBox = function(position) {
  let box = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        vertexColors: THREE.FaceColors,
        transparent: true,
        opacity: 0.5,
      }),
  );

  let outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(box.geometry),
      new THREE.LineBasicMaterial({color: 0xffffff}));

  box.outline = outline;
  box.label = this;

  this.box = box;

  if (this.data) {
    box.position.x = this.data['position'][0];
    box.position.y = this.data['position'][1];
    box.position.z = this.data['position'][2];
    box.outline.position.copy(box.position);

    box.rotation.x = this.data['rotation'][0];
    box.rotation.y = this.data['rotation'][1];
    box.rotation.z = this.data['rotation'][2];
    box.outline.rotation.copy(box.rotation);

    box.scale.x = this.data['scale'][0];
    box.scale.y = this.data['scale'][1];
    box.scale.z = this.data['scale'][2];
    box.outline.scale.copy(box.scale);
  } else {
    box.scale.z = 0.01;

    this.data = {};
    this.data['position'] = [position.x, position.y, position.z];
    this.data['rotation'] = [
      box.rotation.x, box.rotation.y,
      box.rotation.z];
    this.data['scale'] = [box.scale.x, box.scale.y, box.scale.z];

    box.position.copy(position);
    box.outline.position.copy(box.position);
    box.outline.scale.copy(box.scale);
  }

  this.setColor(this.color());

  return box;
};

Box3d.prototype.moveBoxAlongViewPlane = function(
    viewPlaneNormal,
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

  if (this.parent) {
    this.parent.interpolate(this);
  }
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
  this.box.scale.x = Math.abs(boxPlanePoint.x * this.box.scale.x * 2);
  this.box.outline.scale.x = this.box.scale.x;
  this.box.scale.y = Math.abs(boxPlanePoint.y * this.box.scale.y * 2);
  this.box.outline.scale.y = this.box.scale.y;

  if (this.parent) {
    this.parent.interpolate(this);
  }
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
  this.box.scale.z = Math.abs(scale);
  this.box.outline.scale.z = this.box.scale.z;

  if (this.parent) {
    this.parent.interpolate(this);
  }
};

// Rotate box on z axis
Box3d.prototype.rotateBox = function(
    cameraPosition,
    cameraWorldDirection,
    currentProjection,
    newProjection) {
  // Remember that (mouseX, mouseY) must intersect with box
  let newRay = new THREE.Ray(cameraPosition, newProjection);

  let currentRay = new THREE.Ray(cameraPosition, currentProjection);

  let norm = cameraWorldDirection;
  norm.normalize();
  // Note: the constant of THREE.Plane is
  // -(normal.dot(some_point_on_the_plane))
  let viewPlane = new THREE.Plane(norm, -norm.dot(this.box.position));
  let target1 = new THREE.Vector3();
  let currentHit = currentRay.intersectPlane(viewPlane, target1);
  let target2 = new THREE.Vector3();
  let newHit = newRay.intersectPlane(viewPlane, target2);
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

  if (this.parent) {
    this.parent.interpolate(this);
  }
};

Box3d.prototype.toJson = function() {
  let label = this.encodeBaseJson();

  this.data = {};

  this.data.position = [
    this.box.position.x, this.box.position.y,
    this.box.position.z];

  this.data.rotation = [
    this.box.rotation.x, this.box.rotation.y,
    this.box.rotation.z];

  this.data.scale = [
    this.box.scale.x, this.box.scale.y,
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

Box3d.prototype.interpolate = function(prevKeyframeIndex, nextKeyframeIndex,
                                       myIndex, attributes) {
  let distance = nextKeyframeIndex - prevKeyframeIndex;
  let alpha = (myIndex - prevKeyframeIndex) / distance;

  // Interpolate position
  this.box.position.lerpVectors(
      this.parent.children[prevKeyframeIndex].box.position,
      this.parent.children[nextKeyframeIndex].box.position,
      alpha);
  this.box.outline.position.copy(this.box.position);

  // Interpolate size
  this.box.scale.lerpVectors(
      this.parent.children[prevKeyframeIndex].box.scale,
      this.parent.children[nextKeyframeIndex].box.scale,
      alpha);

  this.box.outline.scale.copy(this.box.scale);

  // Interpolate rotation
  THREE.Quaternion.slerp(
      this.parent.children[prevKeyframeIndex].box.quaternion,
      this.parent.children[nextKeyframeIndex].box.quaternion,
      this.box.quaternion,
      alpha);
  this.box.rotation.setFromQuaternion(this.box.quaternion);
  this.box.outline.quaternion.copy(this.box.quaternion);
  this.box.outline.rotation.copy(this.box.rotation);

  this.attributes = attributes;
};

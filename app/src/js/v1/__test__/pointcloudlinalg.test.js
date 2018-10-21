import {Box3dTrack} from '../point_cloud/point_cloud_tracking.js';
import {SatItem, SatLabel} from '../sat.js';
import {Box3d} from '../point_cloud/box3d.js';
import {SatPointCloud} from '../point_cloud/sat_point_cloud.js';
import * as THREE from 'three';
import * as jQuery from 'jquery';

/**
 * Create a mock SatPointCloud Object for testing linear algebra functions
 *  @return {Object}
 */
function createTestPointCloudItem() {
  let pc = new SatItem(null);

  pc.quat = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(0, 1, 0));
  pc.quatInverse = pc.quat.clone().inverse();

  pc.MOUSE_CORRECTION_FACTOR = Math.random();
  pc.MOVE_CORRECTION_FACTOR = Math.random();
  pc.VERTICAL = new THREE.Vector3(0, 0, 1);

  pc.currentView = {};
  pc.currentView.left = Math.random();
  pc.currentView.width = Math.random() * (1.0 - pc.currentView.left);
  pc.currentView.top = Math.random();
  pc.currentView.height = Math.random() * (1.0 - pc.currentView.top);

  pc.currentCamera =
      new THREE.PerspectiveCamera(75, 1.5, 0.1, 1000);
  let random = new THREE.Vector3(Math.random() * 50, Math.random() * 50,
      Math.random() * 50);
  pc.currentCamera.position.copy(random);

  pc.target = new THREE.Vector3(Math.random() * 50,
      Math.random() * 50,
      Math.random() * 50);
  pc.currentCamera.lookAt(pc.target);
  pc.currentCamera.updateProjectionMatrix();

  pc.container = {};
  pc.container.offsetLeft = Math.random() * 500;
  pc.container.offsetTop = Math.random() * 500;
  pc.container.offsetWidth = Math.random() * 500;
  pc.container.offsetHeight = Math.random() * 500;
  pc.container.getBoundingClientRect = function() {
    return {x: pc.container.offsetLeft, y: pc.container.offsetTop,
            width: pc.container.offsetWidth, height: pc.container.offsetHeight,
            top: pc.container.offsetTop, left: pc.container.offsetLeft};
  };

  return pc;
}

/**
 *
 * @param {Box3d} box3d
 * @return {THREE.Vector3} Returns a random point located on the box's surface
 */
function getRandomLocationOnBox(box3d) {
  let face = Math.floor(Math.random() * 6);

  let position = new THREE.Vector3();

  switch (face % 3) {
    case 0:
      position.x += box3d.box.scale.x;
      break;
    case 1:
      position.y += box3d.box.scale.y;
      break;
    case 2:
      position.z += box3d.box.scale.z;
      break;
  }

  if (face % 2 == 1) {
    position.multiplyScalar(-1);
  }

  switch (face % 3) {
    case 0:
      position.y += Math.random() * box3d.box.scale.y -
          box3d.box.scale.y / 2;
      position.z += Math.random() * box3d.box.scale.z -
          box3d.box.scale.z / 2;
      break;
    case 1:
      position.x += Math.random() * box3d.box.scale.x -
          box3d.box.scale.x / 2;
      position.z += Math.random() * box3d.box.scale.z -
          box3d.box.scale.z / 2;
      break;
    case 2:
      position.x += Math.random() * box3d.box.scale.x -
          box3d.box.scale.x / 2;
      position.y += Math.random() * box3d.box.scale.y -
          box3d.box.scale.y / 2;
      break;
  }

  return box3d.box.localToWorld(position);
}

/**
 * get a random number in [a,b)
 * @param {int} a
 * @param {int} b
 * @return {int}
 */
function randomIn(a, b) {
  let x = Math.floor(Math.random() * (b - a));
  return x + a;
}

/**
 * get a random interval in [a,b)
 * @param {int} a
 * @param {int} b
 * @return {[int,int]}
 */
function randomSliceIn(a, b) {
  for (; ;) {
    let x = randomIn(a, b);
    let y = randomIn(a, b);
    if (x != y) {
      if (x > y) {
        [x, y] = [y, x];
      }
      return [x, y];
    }
  }
}

/**
 * Expect two vectors are equal, with k-digit precision
 * @param {THREE.Vector3} a
 * @param {THREE.Vector3} b
 * @param {int} k
 * @return {void}
 */
function expectVectors(a, b, k) {
  expect(a.x).toBeCloseTo(b.x, k);
  expect(a.y).toBeCloseTo(b.y, k);
  expect(a.z).toBeCloseTo(b.z, k);
}

/**
 * Create mock Box3d for testing
 * @return {SatLabel}
 */
function createTestBox3d() {
  let label = new Box3d(null, 0);
  let box = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        vertexColors: THREE.FaceColors,
        transparent: true,
        opacity: 0.5,
      }),
  );

  box.position.copy(new THREE.Vector3(Math.random() * 50, Math.random() * 50,
      Math.random() * 50));
  box.scale.copy(new THREE.Vector3(Math.random() * 50, Math.random() * 50,
      Math.random() * 50));
  box.rotation.copy(new THREE.Euler(Math.random(), Math.random(),
      Math.random(), 'XYZ'));

  let outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(box.geometry),
      new THREE.LineBasicMaterial({color: 0xffffff}));

  outline.position.copy(box.position);
  outline.scale.copy(box.scale);
  outline.rotation.copy(box.rotation);

  box.outline = outline;
  box.label = label;
  label.box = box;

  return label;
}

/**
 *
 * @param {Object} pc
 * @return {float[]} mouse coordinates that lie within pc.currentView
 */
function generateMouseCoordinates(pc) {
  let mX =
      pc.container.offsetLeft + // Put inside container
      pc.container.offsetWidth *
      (pc.currentView.left + Math.random() * pc.currentView.width);
  let mY =
      pc.container.offsetTop +
      pc.container.offsetHeight *
      (pc.currentView.top + Math.random() * pc.currentView.height);
  // pc.container.offsetHeight * pc.currentView.top +
  // Math.random() * pc.currentView.height;
  return [mX, mY];
}

test('forward and left vector calculations', () => {
  let fakePointCloud = createTestPointCloudItem();

  let forward =
      SatPointCloud.prototype.calculateForward.bind(fakePointCloud)();

  // Check that forward is a Vector3 lying in the XY plane
  expect(forward).toBeInstanceOf(THREE.Vector3);
  expect(forward.z).toEqual(0);

  let left =
      SatPointCloud.prototype.calculateLeft.bind(fakePointCloud)(forward);

  // Forward, left, and VERTICAL are all orthogonal
  expect(forward.dot(left)).toBeCloseTo(0, 5);
  expect(forward.dot(fakePointCloud.VERTICAL)).toBeCloseTo(0, 5);
  expect(left.dot(fakePointCloud.VERTICAL)).toBeCloseTo(0, 5);

  forward.normalize();

  let otherForward =
      new THREE.Vector3().subVectors(fakePointCloud.target,
          fakePointCloud.currentCamera.position);
  // Check that forward actually points in the forward direction,
  // implying that left is correct
  otherForward.z = 0;
  otherForward.normalize();
  expect(otherForward.x).toBeCloseTo(forward.x, 5);
  expect(otherForward.y).toBeCloseTo(forward.y, 5);
  expect(otherForward.z).toBeCloseTo(forward.z, 5);
});

test('Mouse projection functions', () => {
  let pc = createTestPointCloudItem();
  pc.currentCamera.getWorldDirection(pc.target);

  let [mX, mY] = generateMouseCoordinates(pc);

  // Calculate NDC
  let [NDCX, NDCY] =
      SatPointCloud.prototype.convertMouseToNDC.bind(pc)(mX, mY);

  pc.convertMouseToNDC = SatPointCloud.prototype.convertMouseToNDC;

  // Check that the coordinates are within [-1, 1]
  expect(NDCX).toBeLessThanOrEqual(1);
  expect(NDCX).toBeGreaterThanOrEqual(-1);
  expect(NDCY).toBeLessThanOrEqual(1);
  expect(NDCY).toBeGreaterThanOrEqual(-1);

  // Get projection
  let projection =
      SatPointCloud.prototype.calculateProjectionFromMouse.bind(pc)(mX, mY);
  // Put projection back into camera space
  projection.add(pc.currentCamera.position);
  projection.project(pc.currentCamera);

  // Check that projection is equal to NDC mouse
  expect(projection.x).toBeCloseTo(NDCX, 5);
  expect(projection.y).toBeCloseTo(NDCY, 5);
});
test('zoom funciton', () => {
  let pc = createTestPointCloudItem();

  let oldposition = new THREE.Vector3();
  oldposition.copy(pc.currentCamera.position);

  let amount = Math.random();// set random value for amount
  SatPointCloud.prototype.zoom.bind(pc)(amount);

  // Check that new Cameraposition is being updated correctly by amount
  expect(pc.currentCamera.position.x).toBeCloseTo(
      (oldposition.x * (1 - amount) + pc.target.x * amount), 5);
  expect(pc.currentCamera.position.y).toBeCloseTo(
      (oldposition.y * (1 - amount) + pc.target.y * amount), 5);
  expect(pc.currentCamera.position.z).toBeCloseTo(
      (oldposition.z * (1 - amount) + pc.target.z * amount), 5);
});

test('Box3d movement', () => {
  // Set up test objects
  let pc = createTestPointCloudItem();
  pc.convertMouseToNDC = SatPointCloud.prototype.convertMouseToNDC;
  let box3d = createTestBox3d();

  // Random mouse projection
  let [mX, mY] = generateMouseCoordinates(pc);
  let projection =
      SatPointCloud.prototype.calculateProjectionFromMouse.bind(pc)(mX, mY);

  // Get random point on box surface
  let boxLocation = getRandomLocationOnBox(box3d);
  let offset = new THREE.Vector3();
  offset.copy(box3d.box.position);
  offset.sub(boxLocation);

  // Random view plane
  let normal = new THREE.Vector3(Math.random(), Math.random(),
      Math.random());
  normal.normalize();

  let originalBoxPosition = new THREE.Vector3();
  originalBoxPosition.copy(box3d.box.position);

  let camToBox = new THREE.Vector3();
  camToBox.copy(pc.currentCamera.position);
  camToBox.sub(boxLocation);

  let dist = -camToBox.dot(normal) /
      projection.dot(normal);
  if (dist < 0) {
    // It means we can't see box in screen
    return;
  }
  expect(dist).toBeGreaterThanOrEqual(0);

  box3d.moveBoxAlongViewPlane(normal, offset, boxLocation,
      pc.currentCamera.position, projection);

  // Check that the box moved along the view plane
  let positionChange = new THREE.Vector3();
  positionChange.subVectors(box3d.box.position, originalBoxPosition);
  expect(positionChange.dot(normal)).toBeCloseTo(0, 5);

  // Check that the box's final position is where the mouse was raycast
  projection =
      SatPointCloud.prototype.calculateProjectionFromMouse.bind(pc)(mX, mY);
  camToBox = new THREE.Vector3();
  camToBox.subVectors(box3d.box.position, pc.currentCamera.position);
  camToBox.sub(offset);
  camToBox.normalize();
  expect(camToBox.x).toBeCloseTo(projection.x);
  expect(camToBox.y).toBeCloseTo(projection.y);
  expect(camToBox.z).toBeCloseTo(projection.z);
});

test('rotate_restricted and rotate_free funciton', () => {
  let pc = createTestPointCloudItem();
  pc.currentCamera.getWorldDirection(pc.target);

  let oldposition = new THREE.Vector3();
  oldposition.copy(pc.currentCamera.position);

  let dx = Math.random() * Math.PI;
  // set random value for dx, ranging from 0-PI
  let dy = Math.random() * 0.5 * Math.PI;
  // set random value for dy, ranging from 0-0.5PI

  SatPointCloud.prototype.rotate_restricted.bind(pc)(dx);

  // check that the cameraposition rotate correctly by dx
  expect(pc.currentCamera.position.x).
      toBeCloseTo((pc.target.x + (oldposition.x - pc.target.x) * Math.cos(dx)
          - (oldposition.y - pc.target.y) * Math.sin(dx)), 5);
  expect(pc.currentCamera.position.y).
      toBeCloseTo((pc.target.y + (oldposition.x - pc.target.x) * Math.sin(dx)
          + (oldposition.y - pc.target.y) * Math.cos(dx)), 5);
  expect(pc.currentCamera.position.z).toBeCloseTo(oldposition.z, 5);
  // check that the coordinate of z remains unchanged

  let newposition = new THREE.Vector3();
  newposition.copy(pc.currentCamera.position);

  SatPointCloud.prototype.rotate_free.bind(pc)(0, dy);

  // Check that after being rotated around axis
  // orthogonal to vertical axis the position is updated by dy
  expect((pc.currentCamera.position.sub(pc.target)).angleTo(
      newposition.sub(pc.target))).toBeCloseTo(dy, 5);
});

test('Box3d rotate', () => {
  // Set up test objects
  let pc = createTestPointCloudItem();
  pc.convertMouseToNDC = SatPointCloud.prototype.convertMouseToNDC.bind(pc);
  let box3d = createTestBox3d();

  let camnorm = pc.currentCamera.getWorldDirection(pc.target);

  let [mX1, mY1] = generateMouseCoordinates(pc);
  let [mX2, mY2] = generateMouseCoordinates(pc);
  let projection1 =
      SatPointCloud.prototype.calculateProjectionFromMouse.bind(pc)(mX1, mY1);
  let projection2 =
      SatPointCloud.prototype.calculateProjectionFromMouse.bind(pc)(mX2, mY2);

  let mX0 = 0.5 * pc.currentView.width + pc.currentView.left;
  mX0 = mX0 * pc.container.offsetWidth + pc.container.offsetLeft;
  let mY0 = 0.5 * pc.currentView.height + pc.currentView.top;
  mY0 = mY0 * pc.container.offsetHeight + pc.container.offsetTop;
  let proj0 =
      SatPointCloud.prototype.calculateProjectionFromMouse.bind(pc)(mX0, mY0);
  expectVectors(proj0, camnorm);

  let ray1 = new THREE.Ray(pc.currentCamera.position, projection1);
  let ray2 = new THREE.Ray(pc.currentCamera.position, projection2);

  let plane = new THREE.Plane(camnorm, -camnorm.dot(box3d.box.position));
  let rel = box3d.box.position.clone();
  rel.sub(pc.currentCamera.position);
  // If we can't see the box at all
  if (camnorm.dot(rel) < 0) {
    return;
  }
  expect(camnorm.dot(rel)).toBeGreaterThanOrEqual(0);
  expect(camnorm.dot(projection1)).toBeGreaterThanOrEqual(0);
  expect(camnorm.dot(projection2)).toBeGreaterThanOrEqual(0);

  let tmp = new THREE.Vector3();
  let hit1 = ray1.intersectPlane(plane, tmp);
  let hit2 = ray2.intersectPlane(plane, tmp);
  expect(hit1).toBeTruthy();
  expect(hit2).toBeTruthy();
  hit1.sub(box3d.box.position);
  hit2.sub(box3d.box.position);
  hit1.normalize();
  hit2.normalize();

  let quat1 = new THREE.Quaternion();
  quat1.clone(box3d.box.quaternion);
  box3d.rotateBox(pc.currentCamera.position,
      camnorm, projection1, projection2);

  let quat2 = new THREE.Quaternion();
  quat2.clone(box3d.box.quaternion);
  hit1.applyQuaternion(quat1.inverse());
  hit1.applyQuaternion(quat2);

  expectVectors(hit1, hit2, 5);
});

/**
 * Interpolate between vector a and b
 * t=0, return a
 * t=1, return b
 * @param {THREE.Vector3} a
 * @param {THREE.Vector3} b
 * @param {float} t
 * @return {THREE.Vector3}
 */
function interpolateVector(a, b, t) {
  let ha = new THREE.Vector3();
  ha.copy(a);
  let hb = new THREE.Vector3();
  hb.copy(b);
  ha.multiplyScalar(1 - t);
  hb.multiplyScalar(t);
  ha.add(hb);
  return ha;
}

test('Box3d interpolate', () => {
  let N = 1000;
  let pc = createTestPointCloudItem();
  pc.children = [];
  for (let i = 0; i < N; i++) {
    let b = createTestBox3d();
    b.parent = pc;
    pc.children.push(b);
  }
  expect(pc.children.length).toBeCloseTo(N);
  let [b, c] = randomSliceIn(0, N);
  let a = randomIn(b, c + 1);
  let bb = jQuery.extend({}, pc.children[b]);
  let bc = jQuery.extend({}, pc.children[c]);
  pc.children[a].interpolate(b, c, a, 0);
  let ba = jQuery.extend({}, pc.children[a]);
  let t = (a - b + 0.0) / (c - b + 0.0);
  let h1 = interpolateVector(bb.box.position, bc.box.position, t);
  expectVectors(h1, ba.box.position, 5);
  let hs = interpolateVector(bb.box.scale, bc.box.scale, t);
  expectVectors(hs, ba.box.scale, 5);
  let q = new THREE.Quaternion();
  q.copy(bb.box.quaternion);
  q.slerp(bc.box.quaternion, t);
  q.normalize();
  let qa = new THREE.Quaternion();
  qa.copy(ba.box.quaternion);
  qa.normalize();
  expect(q.x).toBeCloseTo(qa.x, 5);
  expect(q.y).toBeCloseTo(qa.y, 5);
  expect(q.z).toBeCloseTo(qa.z, 5);
  expect(q.w).toBeCloseTo(qa.w, 5);
});

test('Box3dTrack interpolate', () => {
  const mockInterpolate = jest.fn();
  let N = 1000;
  let bt = new Box3dTrack();
  for (let i = 0; i < N; i++) {
    let b = createTestBox3d();
    b.id = i;
    b.interpolate = mockInterpolate;
    bt.children.push(b);
  }
  expect(bt.children.length).toBe(N);

  // case #1: no prev, no next
  let a = randomIn(0, N);
  let la = new SatLabel();
  let callcnt = 0;
  la.id = a;
  bt.interpolate(la);
  expect(mockInterpolate.mock.calls.length).toBe(callcnt);
  bt.children[a].keyframe = false;

  // case #2: no prev, with next
  a = randomIn(0, N - 1);
  la.id = a;
  let b = randomIn(a + 1, N);
  bt.children[b].keyframe = true;
  bt.interpolate(la);
  callcnt += b - a - 1;
  expect(mockInterpolate.mock.calls.length).toBe(callcnt);
  for (let i = 0; i < b - a - 1; i++) {
    let call = mockInterpolate.mock.calls[i];
    expect(call[0]).toBe(a);
    expect(call[1]).toBe(b);
    expect(call[2]).toBe(a + 1 + i);
  }
  bt.children[a].keyframe = false;
  bt.children[b].keyframe = false;

  // case #3: with prev, no next
  a = randomIn(1, N);
  la.id = a;
  b = randomIn(0, a);
  bt.children[b].keyframe = true;
  bt.interpolate(la);
  let s0 = callcnt;
  callcnt += a - b - 1;
  expect(mockInterpolate.mock.calls.length).toBe(callcnt);
  for (let i = 0; i < a - b - 1; i++) {
    let call = mockInterpolate.mock.calls[s0 + i];
    expect(call[0]).toBe(b);
    expect(call[1]).toBe(a);
    expect(call[2]).toBe(b + 1 + i);
  }
  bt.children[a].keyframe = false;
  bt.children[b].keyframe = false;

  // case #4: with prev, with next
  a = randomIn(1, N - 1);
  la.id = a;
  b = randomIn(0, a);
  let c = randomIn(a + 1, N);
  bt.children[b].keyframe = true;
  bt.children[c].keyframe = true;
  bt.interpolate(la);
  s0 = callcnt;
  callcnt += c - b - 1 - 1;
  expect(mockInterpolate.mock.calls.length).toBe(callcnt);
  for (let i = 0; i < c - b - 1 - 1; i++) {
    let call = mockInterpolate.mock.calls[s0 + i];
    if (i < a - b - 1) {
      expect(call[0]).toBe(b);
      expect(call[1]).toBe(a);
      expect(call[2]).toBe(b + 1 + i);
    } else {
      expect(call[0]).toBe(a);
      expect(call[1]).toBe(c);
      expect(call[2]).toBe(b + 2 + i);
    }
  }
});

import React from 'react';
import Session from '../common/session';
import {PointCloudViewerConfigType} from '../functional/types';
import {withStyles} from '@material-ui/core/styles/index';
import * as THREE from 'three';
import * as types from '../actions/action_types';
import {Object3D} from 'three';
import createStyles from '@material-ui/core/styles/createStyles';

const styles = () => createStyles({
  canvas: {
    position: 'absolute',
    height: '100%',
    width: '100%'
  }
});

interface ClassType {
  /** CSS canvas name */
  canvas: string;
}

interface Props {
  /** CSS class */
  classes: ClassType;
}

/**
 * Get the current item in the state
 * @return {ItemType}
 */
function getCurrentItem() {
  const state = Session.getState();
  return state.items[state.current.item];
}

/**
 * Retrieve the current viewer configuration
 * @return {ViewerConfigType}
 */
function getCurrentViewerConfig() {
  const state = Session.getState();
  return state.items[state.current.item].viewerConfig;
}

/**
 * Canvas Viewer
 */
class PointCloudView extends React.Component<Props> {
  /** Canvas to draw on */
  private canvas: HTMLCanvasElement | null;
  /** ThreeJS Renderer */
  private renderer?: THREE.WebGLRenderer;
  /** ThreeJS Scene object */
  private scene: THREE.Scene;
  /** ThreeJS Camera */
  private camera: THREE.PerspectiveCamera;
  /** ThreeJS sphere mesh for indicating camera target location */
  private target: THREE.Mesh;
  /** ThreeJS raycaster */
  private raycaster: THREE.Raycaster;
  /** Mouse click state */
  private mouseDown: boolean;
  /** Mouse position */
  private mX: number;
  /** Mouse position */
  private mY: number;

  /** Ref Handler */
  private refInitializer:
    (component: HTMLDivElement | HTMLCanvasElement | null) => void;

  /** UI handler */
  private mouseDownHandler: (e: React.MouseEvent<HTMLCanvasElement>) => void;
  /** UI handler */
  private mouseUpHandler: (e: React.MouseEvent<HTMLCanvasElement>) => void;
  /** UI handler */
  private mouseMoveHandler: (e: React.MouseEvent<HTMLCanvasElement>) => void;
  /** UI handler */
  private keyDownHandler: (e: KeyboardEvent) => void;
  /** UI handler */
  private mouseWheelHandler: (e: React.WheelEvent<HTMLCanvasElement>) => void;
  /** UI handler */
  private doubleClickHandler: () => void;

  /** Factor to divide mouse delta by */
  private MOUSE_CORRECTION_FACTOR: number;
  /** Move amount when using arrow keys */
  private MOVE_AMOUNT: number;
  // private UP_KEY: number;
  // private DOWN_KEY: number;
  // private LEFT_KEY: number;
  // private RIGHT_KEY: number;
  // private PERIOD_KEY: number;
  // private SLASH_KEY: number;
  /**
   * Constructor, handles subscription to store
   * @param {Object} props: react props
   */
  constructor(props: Readonly<Props>) {
    super(props);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
    this.target = new THREE.Mesh(
      new THREE.SphereGeometry(0.03),
        new THREE.MeshBasicMaterial({
          color:
            0xffffff
        }));
    this.scene.add(this.target);

    this.raycaster = new THREE.Raycaster();
    this.raycaster.linePrecision = 0.5;
    this.raycaster.near = 1.0;
    this.raycaster.far = 100.0;

    this.canvas = null;

    this.mouseDown = false;
    this.mX = 0;
    this.mY = 0;

    this.refInitializer = this.initializeRefs.bind(this);

    this.mouseDownHandler = this.handleMouseDown.bind(this);
    this.mouseUpHandler = this.handleMouseUp.bind(this);
    this.mouseMoveHandler = this.handleMouseMove.bind(this);
    this.keyDownHandler = this.handleKeyDown.bind(this);
    this.mouseWheelHandler = this.handleMouseWheel.bind(this);
    this.doubleClickHandler = this.handleDoubleClick.bind(this);

    this.MOUSE_CORRECTION_FACTOR = 80.0;
    this.MOVE_AMOUNT = 0.3;

    // this.UP_KEY = 38;
    // this.DOWN_KEY = 40;
    // this.LEFT_KEY = 37;
    // this.RIGHT_KEY = 39;
    // this.PERIOD_KEY = 190;
    // this.SLASH_KEY = 191;

    document.addEventListener('keydown', this.keyDownHandler);
  }

  /**
   * Normalize mouse coordinates
   * @param {number} mX: Mouse x-coord
   * @param {number} mY: Mouse y-coord
   * @return {Array<number>}
   */
  private convertMouseToNDC(mX: number, mY: number): number[] {
    if (this.canvas) {
      let x = mX / this.canvas.offsetWidth;
      let y = mY / this.canvas.offsetHeight;
      x = 2 * x - 1;
      y = -2 * y + 1;

      return [x, y];
    }
    return [0, 0];
  }

  /**
   * Handle mouse down
   * @param {React.MouseEvent<HTMLCanvasElement>} e
   */
  private handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    e.stopPropagation();
    this.mouseDown = true;
  }

  /**
   * Handle mouse up
   * @param {React.MouseEvent<HTMLCanvasElement>} e
   */
  private handleMouseUp(e: React.MouseEvent<HTMLCanvasElement>) {
    e.stopPropagation();
    this.mouseDown = false;
  }

  /**
   * Handle mouse move
   * @param {React.MouseEvent<HTMLCanvasElement>} e
   */
  private handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    e.stopPropagation();

    if (!this.canvas) {
      return;
    }

    const newX = e.clientX -
      this.canvas.getBoundingClientRect().left;
    const newY = e.clientY -
      this.canvas.getBoundingClientRect().top;

    if (this.mouseDown) {
      const viewerConfig: PointCloudViewerConfigType =
        getCurrentViewerConfig() as PointCloudViewerConfigType;

      const target = new THREE.Vector3(viewerConfig.target.x,
        viewerConfig.target.y,
        viewerConfig.target.z);
      const offset = new THREE.Vector3(viewerConfig.position.x,
        viewerConfig.position.y,
        viewerConfig.position.z);
      offset.sub(target);

      // Rotate so that positive y-axis is vertical
      const rotVertQuat = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(viewerConfig.verticalAxis.x,
          viewerConfig.verticalAxis.y,
          viewerConfig.verticalAxis.z),
        new THREE.Vector3(0, 1, 0));
      offset.applyQuaternion(rotVertQuat);

      // Convert to spherical coordinates
      const spherical = new THREE.Spherical();
      spherical.setFromVector3(offset);

      // Apply rotations
      spherical.theta += (newX - this.mX) / this.MOUSE_CORRECTION_FACTOR;
      spherical.phi += (newY - this.mY) / this.MOUSE_CORRECTION_FACTOR;

      spherical.phi = Math.max(0, Math.min(Math.PI, spherical.phi));

      spherical.makeSafe();

      // Convert to Cartesian
      offset.setFromSpherical(spherical);

      // Rotate back to original coordinate space
      const quatInverse = rotVertQuat.clone().inverse();
      offset.applyQuaternion(quatInverse);

      offset.add(target);

      Session.dispatch({
        type: types.MOVE_CAMERA,
        newPosition: {x: offset.x, y: offset.y, z: offset.z}
      });
    }

    this.mX = newX;
    this.mY = newY;
  }

  /**
   * Handle keyboard events
   * @param {KeyboardEvent} e
   */
  private handleKeyDown(e: KeyboardEvent) {
    const viewerConfig: PointCloudViewerConfigType =
      getCurrentViewerConfig() as PointCloudViewerConfigType;

    // Get vector pointing from camera to target projected to horizontal plane
    let forwardX = viewerConfig.target.x - viewerConfig.position.x;
    let forwardY = viewerConfig.target.y - viewerConfig.position.y;
    const forwardDist = Math.sqrt(forwardX * forwardX + forwardY * forwardY);
    forwardX *= this.MOVE_AMOUNT / forwardDist;
    forwardY *= this.MOVE_AMOUNT / forwardDist;
    const forward = new THREE.Vector3(forwardX, forwardY, 0);

    // Get vector pointing up
    const vertical = new THREE.Vector3(
      viewerConfig.verticalAxis.x,
      viewerConfig.verticalAxis.y,
      viewerConfig.verticalAxis.z
    );

    // Handle movement in three dimensions
    const left = new THREE.Vector3();
    left.crossVectors(vertical, forward);
    left.normalize();
    left.multiplyScalar(this.MOVE_AMOUNT);

    switch (e.key) {
      case '.':
        Session.dispatch({
          type: types.MOVE_CAMERA_AND_TARGET,
          newPosition: {
            x: viewerConfig.position.x,
            y: viewerConfig.position.y,
            z: viewerConfig.position.z + this.MOVE_AMOUNT
          },
          newTarget: {
            x: viewerConfig.target.x,
            y: viewerConfig.target.y,
            z: viewerConfig.target.z + this.MOVE_AMOUNT
          }
        });
        break;
      case '/':
        Session.dispatch({
          type: types.MOVE_CAMERA_AND_TARGET,
          newPosition: {
            x: viewerConfig.position.x,
            y: viewerConfig.position.y,
            z: viewerConfig.position.z - this.MOVE_AMOUNT
          },
          newTarget: {
            x: viewerConfig.target.x,
            y: viewerConfig.target.y,
            z: viewerConfig.target.z - this.MOVE_AMOUNT
          }
        });
        break;
      case 'Down':
      case 'ArrowDown':
        Session.dispatch({
          type: types.MOVE_CAMERA_AND_TARGET,
          newPosition: {
            x: viewerConfig.position.x - forwardX,
            y: viewerConfig.position.y - forwardY,
            z: viewerConfig.position.z
          },
          newTarget: {
            x: viewerConfig.target.x - forwardX,
            y: viewerConfig.target.y - forwardY,
            z: viewerConfig.target.z
          }
        });
        break;
      case 'Up':
      case 'ArrowUp':
        Session.dispatch({
          type: types.MOVE_CAMERA_AND_TARGET,
          newPosition: {
            x: viewerConfig.position.x + forwardX,
            y: viewerConfig.position.y + forwardY,
            z: viewerConfig.position.z
          },
          newTarget: {
            x: viewerConfig.target.x + forwardX,
            y: viewerConfig.target.y + forwardY,
            z: viewerConfig.target.z
          }
        });
        break;
      case 'Left':
      case 'ArrowLeft':
        Session.dispatch({
          type: types.MOVE_CAMERA_AND_TARGET,
          newPosition: {
            x: viewerConfig.position.x + left.x,
            y: viewerConfig.position.y + left.y,
            z: viewerConfig.position.z + left.z
          },
          newTarget: {
            x: viewerConfig.target.x + left.x,
            y: viewerConfig.target.y + left.y,
            z: viewerConfig.target.z + left.z
          }
        });
        break;
      case 'Right':
      case 'ArrowRight':
        Session.dispatch({
          type: types.MOVE_CAMERA_AND_TARGET,
          newPosition: {
            x: viewerConfig.position.x - left.x,
            y: viewerConfig.position.y - left.y,
            z: viewerConfig.position.z - left.z
          },
          newTarget: {
            x: viewerConfig.target.x - left.x,
            y: viewerConfig.target.y - left.y,
            z: viewerConfig.target.z - left.z
          }
        });
        break;
    }
  }

  /**
   * Handle mouse wheel
   * @param {React.WheelEvent<HTMLCanvasElement>} e
   */
  private handleMouseWheel(e: React.WheelEvent<HTMLCanvasElement>) {
    const viewerConfig: PointCloudViewerConfigType =
      getCurrentViewerConfig() as PointCloudViewerConfigType;

    const target = new THREE.Vector3(viewerConfig.target.x,
      viewerConfig.target.y,
      viewerConfig.target.z);
    const offset = new THREE.Vector3(viewerConfig.position.x,
      viewerConfig.position.y,
      viewerConfig.position.z);
    offset.sub(target);

    const spherical = new THREE.Spherical();
    spherical.setFromVector3(offset);

    // Decrease distance from origin by amount specified
    const amount = e.deltaY / this.MOUSE_CORRECTION_FACTOR;
    const newRadius = (1 - amount) * spherical.radius;
    // Limit zoom to not be too close
    if (newRadius > 0.1) {
      spherical.radius = newRadius;

      offset.setFromSpherical(spherical);

      offset.add(target);

      Session.dispatch({
        type: types.MOVE_CAMERA,
        newPosition: {x: offset.x, y: offset.y, z: offset.z}
      });
    }
  }

  /**
   * Handle double click
   */
  private handleDoubleClick() {
    const NDC = this.convertMouseToNDC(
      this.mX,
      this.mY);
    const x = NDC[0];
    const y = NDC[1];

    this.raycaster.setFromCamera(new THREE.Vector2(x, y), this.camera);
    const item = getCurrentItem();
    const pointCloud = Session.pointClouds[item.index];

    const intersects = this.raycaster.intersectObject(pointCloud);

    if (intersects.length > 0) {
      const newTarget = intersects[0].point;
      const viewerConfig: PointCloudViewerConfigType =
        getCurrentViewerConfig() as PointCloudViewerConfigType;
      Session.dispatch({
        type: types.MOVE_CAMERA_AND_TARGET,
        newPosition: {
          x: viewerConfig.position.x - viewerConfig.target.x + newTarget.x,
          y: viewerConfig.position.y - viewerConfig.target.y + newTarget.y,
          z: viewerConfig.position.z - viewerConfig.target.z + newTarget.z
        },
        newTarget: {
          x: newTarget.x,
          y: newTarget.y,
          z: newTarget.z
        }
      });
    }
  }

  /**
   * Set references to div elements and try to initialize renderer
   * @param {HTMLDivElement} component
   * @param {string} componentType
   */
  private initializeRefs(component:
                               HTMLDivElement | HTMLCanvasElement | null) {
    if (!component) {
      return;
    }

    if (component.nodeName === 'CANVAS') {
      this.canvas = component as HTMLCanvasElement;
    }

    if (this.canvas) {
      const rendererParams = {canvas: this.canvas};
      this.renderer = new THREE.WebGLRenderer(rendererParams);
      if (getCurrentItem().loaded) {
        this.updateRenderer();
      }
    }
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  public render() {
    const {classes} = this.props;
    return (
        <canvas className={classes.canvas} ref={this.refInitializer}
          onMouseDown={this.mouseDownHandler} onMouseUp={this.mouseUpHandler}
          onMouseMove={this.mouseMoveHandler} onWheel={this.mouseWheelHandler}
          onDoubleClick={this.doubleClickHandler}
        />
    );
  }

  /**
   * Update rendering constants
   */
  private updateRenderer() {
    const config: PointCloudViewerConfigType =
      getCurrentViewerConfig() as PointCloudViewerConfigType;
    this.target.position.x = config.target.x;
    this.target.position.y = config.target.y;
    this.target.position.z = config.target.z;

    if (this.canvas) {
      this.camera.aspect = this.canvas.offsetWidth /
        this.canvas.offsetHeight;
      this.camera.updateProjectionMatrix();
    }

    this.camera.up.x = config.verticalAxis.x;
    this.camera.up.y = config.verticalAxis.y;
    this.camera.up.z = config.verticalAxis.z;
    this.camera.position.x = config.position.x;
    this.camera.position.y = config.position.y;
    this.camera.position.z = config.position.z;
    this.camera.lookAt(this.target.position);

    if (this.renderer && this.canvas) {
      this.renderer.setSize(this.canvas.offsetWidth,
        this.canvas.offsetHeight);
    }
  }

  /**
   * Execute when component state is updated
   */
  public componentDidUpdate() {
    this.redraw();
  }

  /**
   * Handles canvas redraw
   * @return {boolean}
   */
  public redraw(): boolean {
    const state = Session.getState();
    const item = state.current.item;
    const loaded = state.items[item].loaded;
    if (loaded) {
      const pointCloud = Session.pointClouds[item];
      if (this.scene.children.length !== 1) {
        this.scene.children = [new Object3D()];
      }
      if (this.scene.children[0] !== pointCloud) {
        this.scene.children[0] = pointCloud;
      }

      if (this.canvas) {
        this.updateRenderer();
      }

      if (this.renderer) {
        this.renderer.render(this.scene, this.camera);
      }
    }
    return true;
  }
}

export default withStyles(styles, {withTheme: true})(PointCloudView);

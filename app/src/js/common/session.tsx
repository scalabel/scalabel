// @flow
import {sprintf} from 'sprintf-js';
import * as types from '../actions/action_types';
import {ImageViewerConfigType, PointCloudViewerConfigType,
  StateType} from '../functional/types';
import _ from 'lodash';
import {makeImageViewerConfig,
  makePointCloudViewerConfig} from '../functional/states';
import {configureStore, configureFastStore} from './configure_store';
import {WindowType} from './window';
import * as THREE from 'three';
import {PLYLoader} from '../thirdparty/PLYLoader';

/**
 * Singleton session class
 */
class Session {
  /** The store to save states */
  public store: any;
  /** The store to save fast states */
  public fastStore: any; // This store contains the temporary state
  /** Images of the session */
  public images: HTMLImageElement[];
  /** Point cloud */
  public pointClouds: THREE.Points[];
  /** Item type: image, point cloud */
  public itemType: string;
  /** Label type: bounding box, segmentation */
  public labelType: string;
  /** The window component */
  public window?: WindowType;
  /** Dev mode */
  public devMode: boolean;

  /**
   * no-op for state initialization
   */
  constructor() {
    this.store = {};
    this.fastStore = configureFastStore();
    this.images = [];
    this.pointClouds = [];
    this.itemType = '';
    this.labelType = '';
    // TODO: make it configurable in the url
    this.devMode = true;
  }

  /**
   * Get current state in store
   * @return {StateType}
   */
  public getState(): StateType {
    return this.store.getState().present;
  }

  /**
   * Get the current temporary state. It is for animation rendering.
   * @return {StateType}
   */
  public getFastState(): StateType {
    return this.fastStore.getState();
  }

  /**
   * Wrapper for redux store dispatch
   * @param {any} action: action description
   */
  public dispatch(action: any): void {
    this.store.dispatch(action);
  }

  /**
   * Subscribe all the controllers to the states
   * @param {any} component: view component
   */
  public subscribe(component: any) {
    if (this.store.subscribe) {
      this.store.subscribe(component.onStateUpdated.bind(component));
    }
    // this.fastStore.subscribe(c.onFastStateUpdated.bind(c));
  }

  /**
   * Initialize state store
   * @param {any} stateJson: json state from backend
   */
  public initStore(stateJson: any): void {
    this.store = configureStore(stateJson, this.devMode);
    this.store.dispatch({type: types.INIT_SESSION});
    (window as any).store = this.store;
    const state = this.getState();
    this.itemType = state.config.itemType;
    this.labelType = state.config.labelType;
  }

  /**
   * Load all the images in the state
   */
  private loadImages(): void {
    const self = this;
    const items = this.getState().items;
    for (const item of items) {
      // Copy item config
      let config: ImageViewerConfigType = {
        ...(item.viewerConfig as ImageViewerConfigType)};
      if (_.isEmpty(config)) {
        config = makeImageViewerConfig();
      }
      const url = item.url;
      const image = new Image();
      image.crossOrigin = 'Anonymous';
      self.images.push(image);
      image.onload = function() {
        config.imageHeight = (this as any).height;
        config.imageWidth = (this as any).width;
        self.store.dispatch({type: types.LOAD_ITEM, index: item.index,
          config});
      };
      image.onerror = function() {
        alert(sprintf('Image %s was not found.', url));
      };
      image.src = url;
    }
  }

  /**
   * Load all point clouds in state
   */
  private loadPointClouds(): void {
    const self = this;
    const loader = new PLYLoader();
    const vertexShader =
      `
        varying float distFromOrigin;
        void main() {
          vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
          distFromOrigin = length(position);
          gl_PointSize = 0.1 * ( 300.0 / -mvPosition.z );
          gl_Position = projectionMatrix * mvPosition;
        }
      `;
    const fragmentShader =
      `
        varying float distFromOrigin;
        uniform vec3 red;
        uniform vec3 yellow;
        uniform vec3 green;
        uniform vec3 teal;
        vec3 getHeatMapColor(float dist) {
          if (dist < 8.0) {
            float val = dist / 8.0;
            return (1.0 - val) * red + val * yellow;
          } else if (dist < 16.0) {
            float val = (dist - 8.0) / 8.0;
            return (1.0 - val) * yellow + val * green;
          } else {
            float val = (dist - 16.0) / 8.0;
            return (1.0 - val) * green + val * teal;
          }
        }
        void main() {
          gl_FragColor = vec4(getHeatMapColor(distFromOrigin), 1.0);
        }
      `;

    const items = this.getState().items;
    for (const item of items) {
      let config: PointCloudViewerConfigType = {
        ...(item.viewerConfig as PointCloudViewerConfigType)};
      if (_.isEmpty(config)) {
        config = makePointCloudViewerConfig();
      }
      loader.load(item.url, function(geometry: THREE.BufferGeometry) {
          const material = new THREE.ShaderMaterial({
            uniforms: {
              red: {
                value: new THREE.Color(0xff0000)
              },
              yellow: {
                value: new THREE.Color(0xffff00)
              },
              green: {
                value: new THREE.Color(0x00ff00)
              },
              teal: {
                value: new THREE.Color(0x00ffff)
              }
            },
            vertexShader,
            fragmentShader,
            alphaTest: 1.0
          });

          const particles = new THREE.Points(geometry, material);
          self.pointClouds.push(particles);

          self.store.dispatch({type: types.LOAD_ITEM, index: item.index,
            config});
        },

        () => null,

        function() {
          alert('Point cloud at ' + item.url + ' was not found.');
        }
      );
    }
  }

  /**
   * Load labeling data initialization function
   */
  public loadData(): void {
    if (this.itemType === 'image') {
      this.loadImages();
    } else if (this.itemType === 'pointcloud') {
      this.loadPointClouds();
    }
  }
}

export default new Session();

import _ from 'lodash'
import { sprintf } from 'sprintf-js'
import * as THREE from 'three'
import * as types from '../action/types'
import { Window } from '../components/window'
import {
  makeImageViewerConfig,
  makePointCloudViewerConfig
} from '../functional/states'
import {
  ImageViewerConfigType, PointCloudViewerConfigType,
  State
} from '../functional/types'
import { PLYLoader } from '../thirdparty/PLYLoader'
import { configureFastStore, configureStore } from './configure_store'

/**
 * Singleton session class
 */
class Session {
  /** The store to save states */
  public store: any
  /** The store to save fast states */
  public fastStore: any // This store contains the temporary state
  /** Images of the session */
  public images: HTMLImageElement[]
  /** Point cloud */
  public pointClouds: THREE.Points[]
  /** Item type: image, point cloud */
  public itemType: string
  /** Label type: bounding box, segmentation */
  public labelType: string
  /** The window component */
  public window?: Window
  /** Dev mode */
  public devMode: boolean

  /**
   * no-op for state initialization
   */
  constructor () {
    this.store = {}
    this.fastStore = configureFastStore()
    this.images = []
    this.pointClouds = []
    this.itemType = ''
    this.labelType = ''
    // TODO: make it configurable in the url
    this.devMode = true
    this.setListeners()
  }

  /**
   * Get current state in store
   * @return {State}
   */
  public getState (): State {
    return this.store.getState().present
  }

  /**
   * Get the current temporary state. It is for animation rendering.
   * @return {State}
   */
  public getFastState (): State {
    return this.fastStore.getState()
  }

  /**
   * Wrapper for redux store dispatch
   * @param {any} action: action description
   */
  public dispatch (action: any): void {
    this.store.dispatch(action)
  }

  /**
   * Subscribe all the controllers to the states
   * @param {Function} callback: view component
   */
  public subscribe (callback: () => void) {
    this.store.subscribe(callback)
    this.fastStore.subscribe(callback)
  }

  /**
   * Initialize state store
   * @param {any} stateJson: json state from backend
   */
  public initStore (stateJson: any): void {
    this.store = configureStore(stateJson, this.devMode)
    this.dispatch({ type: types.INIT_SESSION });
    (window as any).store = this.store
    const state = this.getState()
    this.itemType = state.config.itemType
    this.labelType = state.config.labelType
  }

  /**
   * Load labeling data initialization function
   */
  public loadData (): void {
    if (this.itemType === 'image') {
      this.loadImages()
    } else if (this.itemType === 'pointcloud') {
      this.loadPointClouds()
    }
  }

  /**
   * set listeners for the session
   */
  private setListeners () {
    const body = document.getElementsByTagName('BODY') as
      HTMLCollectionOf<HTMLElement>
    body[0].onresize = () => {
      this.dispatch({ type: types.UPDATE_ALL })
    }
  }

  /**
   * Load all the images in the state
   */
  private loadImages (): void {
    const items = this.getState().items
    const self = this
    for (const item of items) {
      // Copy item config
      let config: ImageViewerConfigType = {
        ...(item.viewerConfig as ImageViewerConfigType)
      }
      if (_.isEmpty(config)) {
        config = makeImageViewerConfig()
      }
      const url = item.url
      const image = new Image()
      image.crossOrigin = 'Anonymous'
      this.images.push(image)
      image.onload = () => {
        config.imageHeight = image.height
        config.imageWidth = image.width
        self.dispatch({
          type: types.LOAD_ITEM, index: item.index,
          config
        })
      }
      image.onerror = () => {
        alert(sprintf('Failed to load image at %s', url))
      }
      image.src = url
    }
  }

  /**
   * Load all point clouds in state
   */
  private loadPointClouds (): void {
    const self = this
    const loader = new PLYLoader()
    const vertexShader =
      `
        varying float distFromOrigin;
        void main() {
          vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
          distFromOrigin = length(position);
          gl_PointSize = 0.1 * ( 300.0 / -mvPosition.z );
          gl_Position = projectionMatrix * mvPosition;
        }
      `
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
      `

    const items = this.getState().items
    for (const item of items) {
      let config: PointCloudViewerConfigType = {
        ...(item.viewerConfig as PointCloudViewerConfigType)
      }
      if (_.isEmpty(config)) {
        config = makePointCloudViewerConfig()
      }
      // tslint:disable-next-line
      loader.load(item.url, (geometry: THREE.BufferGeometry) => {

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
        })

        const particles = new THREE.Points(geometry, material)
        self.pointClouds.push(particles)

        self.store.dispatch({
          type: types.LOAD_ITEM, index: item.index,
          config
        })
      },

        () => null,

        () => {
          alert('Point cloud at ' + item.url + ' was not found.')
        }
      )
    }
  }
}

export default new Session()

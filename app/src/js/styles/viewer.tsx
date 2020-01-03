import { createStyles } from '@material-ui/core'

export const viewerStyles = () => createStyles({
  viewer_container: {
    'display': 'block',
    'height': '100%',
    'position': 'absolute',
    'overflow': 'scroll',
    'outline': 'none',
    'width': '100%',
    'touch-action': 'none'
  },
  camera_button: {
    'color': '#ced4da',
    'z-index': 1001
  },
  camera_y_lock_icon: {
    'z-index': 1000
  },
  camera_x_lock_icon: {
    'color': '#ced4da',
    'transform': 'rotate(90deg)',
    'z-index': 1000,
    'padding-top': '5px'
  }
})

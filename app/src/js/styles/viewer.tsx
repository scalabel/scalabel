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
  }
})

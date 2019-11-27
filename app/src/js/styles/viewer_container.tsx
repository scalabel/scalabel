import { createStyles } from '@material-ui/core'

export const viewerContainerStyles = () => createStyles({
  viewer_container_bar: {
    position: 'absolute',
    zIndex: 10
  },
  select: {
    'backgroundColor': 'rgba(34, 34, 34, 1)',
    'border': '1px solid #ced4da',
    'color': '#ced4da',
    'borderRadius': 4,
    'padding': '10px 26px 10px 12px',
    '&:focus': {
      borderRadius: 4
    },
    'margin-right': '5px'
  },
  icon: {
    color: '#ced4da'
  },
  icon90: {
    color: '#ced4da',
    transform: 'rotate(90deg)'
  },
  viewer_container: {
    display: 'block',
    height: '100%',
    position: 'absolute',
    overflow: 'hidden',
    outline: 'none',
    width: '100%'
  }
})

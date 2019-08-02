import { Theme } from '@material-ui/core/styles'
import createStyles from '@material-ui/core/styles/createStyles'
import { defaultAppBar, defaultHeader } from './general'

// styles used in the create and dashboard navigation page
export const drawerWidth = 240

export const dividedPageStyle = (theme: Theme) => createStyles({
  root: {
    display: 'flex',
    alignItems: 'left'
  },
  appBar: {
    ...defaultAppBar,
    zIndex: theme.zIndex.drawer + 1
  },

  drawer: {
    width: drawerWidth,
    flexShrink: 0
  },

  drawerPaper: {
    width: drawerWidth,
    background: theme.palette.secondary.main
  },

  drawerHeader: {
    ...defaultHeader
  },

  content: {
    flexGrow: 1
  }
})

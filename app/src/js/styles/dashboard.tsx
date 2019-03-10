import createStyles from '@material-ui/core/styles/createStyles';

/* dashboardStyles */
export const dashboardStyles: any = (theme: any) =>  createStyles({
    root: {
        display: 'flex'
    },
    toolbar: {
        paddingRight: 24
    },
    toolbarIcon: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end',
        padding: '0 8px',
        ...theme.mixins.toolbar
    },
    appBar: {
        backgroundColor: '#333',
        zIndex: theme.zIndex.drawer + 1,
        transition: theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen
        })
    },
    title: {
        flexGrow: 1
    },
    drawerPaper: {
        position: 'relative',
        whiteSpace: 'nowrap',
        width: 285,
        transition: theme.transitions.create('width', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen
        })
    },
    appBarSpacer: theme.mixins.toolbar,
    content: {
        flexGrow: 1,
        padding: theme.spacing.unit * 3,
        height: '100vh',
        overflow: 'auto'
    },
    chartContainer: {
        marginLeft: -22
    },
    h5: {
        marginBottom: theme.spacing.unit * 2
    }
});

/* tableStyles */
export const tableStyles: any = (theme: any) =>  createStyles({
    root: {
        width: '100%',
        marginTop: theme.spacing.unit * 3,
        overflowX: 'auto'
    },
    table: {
        minWidth: 700
    },
    row: {
        '&:nth-of-type(odd)': {
            backgroundColor: theme.palette.background.default
        }
    }
});

/* tableCellStyles */
export const tableCellStyles: any = (theme: any) =>  createStyles({
    head: {
        backgroundColor: '#333',
        color: theme.palette.common.white
    },
    body: {
        fontSize: 16
    }
});

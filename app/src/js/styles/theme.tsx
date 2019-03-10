import createMuiTheme, {ThemeOptions} from '@material-ui/core/styles/createMuiTheme';
import {Palette, PaletteOptions} from '@material-ui/core/styles/createPalette';

declare module '@material-ui/core/styles/createMuiTheme' {
    interface Theme {
        palette: Palette;
    }
    interface ThemeOptions {
        palette?: PaletteOptions;
    }
}

export default function createMyTheme(options: ThemeOptions) {
    return createMuiTheme({
        palette: {
            primary: {
                main: '#616161'
            }
        }
    });
}

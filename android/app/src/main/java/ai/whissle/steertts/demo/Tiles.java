package ai.whissle.steertts.demo;

import android.app.PendingIntent;
import android.content.Intent;
import android.os.Build;
import android.service.quicksettings.Tile;
import android.service.quicksettings.TileService;

/**
 * Quick Settings tiles, for SteerTTS without opening the app: "Read copied text" (ClipReadActivity reads the clipboard
 * into the player), "Read this screen" (ScreenReader) and "Catch me up" (the messages that waited, ReadoutListener.catchUp).
 * Private content waits for the phone to be unlocked; a second tap on a tile stops what it started.
 */
public final class Tiles {
    private Tiles() {}

    public static class ReadCopied extends TileService {
        @Override public void onStartListening() {
            Tile t = getQsTile();
            if (t == null) return;
            t.setState(Tile.STATE_INACTIVE);
            t.updateTile();
        }

        @Override public void onClick() {
            Intent i = new Intent(this, ClipReadActivity.class).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            if (Build.VERSION.SDK_INT >= 34)
                startActivityAndCollapse(PendingIntent.getActivity(this, 0, i, PendingIntent.FLAG_IMMUTABLE));
            else startActivityAndCollapseCompat(i);
        }

        @SuppressWarnings("deprecation")
        private void startActivityAndCollapseCompat(Intent i) { startActivityAndCollapse(i); }
    }

    /** Reads the app underneath (ScreenReader); before it is switched on, opens the setup in the app. */
    public static class ReadScreen extends TileService {
        @Override public void onStartListening() {
            Tile t = getQsTile();
            if (t == null) return;
            ScreenReader r = ScreenReader.instance;
            boolean on = r != null && r.isReading();
            t.setState(on ? Tile.STATE_ACTIVE : Tile.STATE_INACTIVE);
            if (Build.VERSION.SDK_INT >= 29) t.setSubtitle(!ScreenReader.enabled() ? "Set up" : on ? "Tap to stop" : "Tap to listen");
            t.updateTile();
        }

        @Override public void onClick() {
            ScreenReader r = ScreenReader.instance;
            if (r != null && r.isReading()) {   // a second tap stops
                r.stop();
                onStartListening();
                return;
            }
            if (r != null) {
                if (isLocked()) unlockAndRun(r::readAfterShade);   // the page to read is behind the lock screen
                else r.readAfterShade();
                return;
            }
            Intent i = new Intent(this, MainActivity.class).putExtra("tab", "assistant").putExtra("setup_screen", true)
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
            if (Build.VERSION.SDK_INT >= 34)
                startActivityAndCollapse(PendingIntent.getActivity(this, 1, i, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT));
            else start(i);
        }

        @SuppressWarnings("deprecation")
        private void start(Intent i) { startActivityAndCollapse(i); }
    }

    public static class CatchUp extends TileService {
        @Override public void onStartListening() {
            Tile t = getQsTile();
            if (t == null) return;
            int n = Digest.size();
            t.setState(n > 0 ? Tile.STATE_ACTIVE : Tile.STATE_INACTIVE);
            if (Build.VERSION.SDK_INT >= 29) t.setSubtitle(n > 0 ? n + " waiting" : "All caught up");
            t.updateTile();
        }

        @Override public void onClick() {
            Announcer a = Announcer.get(this);
            if (a.speaking()) {   // a second tap stops whatever the assistant is saying
                a.stop();
                onStartListening();
                return;
            }
            // the messages are private: never read from the lock screen, unlock first
            Runnable go = () -> {
                ReadoutListener.catchUp(this, true);
                onStartListening();
            };
            if (isLocked()) unlockAndRun(go);
            else go.run();
        }
    }
}

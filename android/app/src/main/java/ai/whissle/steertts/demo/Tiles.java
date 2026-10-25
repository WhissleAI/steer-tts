package ai.whissle.steertts.demo;

import android.app.PendingIntent;
import android.content.Intent;
import android.os.Build;
import android.service.quicksettings.Tile;
import android.service.quicksettings.TileService;

/**
 * Quick Settings tiles, for SteerTTS without opening the app: "Read copied text" (ClipReadActivity reads the clipboard
 * into the player) and "Catch me up" (the messages that waited, ReadoutListener.catchUp).
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
            ReadoutListener.catchUp(this, true);
            onStartListening();
        }
    }
}

package ai.whissle.steertts.demo;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.media.MediaMetadata;
import android.media.session.MediaSession;
import android.media.session.PlaybackState;
import android.os.Build;
import android.os.IBinder;

/**
 * Keeps reading aloud alive outside the app and puts it where media controls live: a media-playback foreground
 * service with a MediaSession (lock screen, notification shade, Bluetooth and headset buttons) and a media
 * notification with previous / play-pause / next / close. Narrator calls update() whenever its state changes.
 */
public class ReaderService extends Service implements Narrator.Listener {
    private static final String CHANNEL = "reading", ACTION = "ai.whissle.steertts.READER";
    private static final int ID = 2;
    private MediaSession session;
    private Narrator narrator;

    static void update(Context c) {
        Narrator n = Narrator.get(c);
        if (n.doc() == null) return;
        Intent i = new Intent(c, ReaderService.class);
        if (n.playing()) c.startForegroundService(i);
        else c.startService(i);
    }

    static void stop(Context c) { c.stopService(new Intent(c, ReaderService.class)); }

    @Override public IBinder onBind(Intent i) { return null; }

    @Override public void onCreate() {
        super.onCreate();
        narrator = Narrator.get(this);
        session = new MediaSession(this, "SteerTTS");
        session.setCallback(new MediaSession.Callback() {
            @Override public void onPlay() { narrator.play(); }
            @Override public void onPause() { narrator.pause(); }
            @Override public void onSkipToNext() { narrator.next(); }
            @Override public void onSkipToPrevious() { narrator.prev(); }
            @Override public void onStop() { narrator.close(); }
            @Override public void onSeekTo(long pos) { narrator.seek((int) (pos / 1000)); }
        });
        session.setSessionActivity(PendingIntent.getActivity(this, 0, new Intent(this, ReaderActivity.class),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT));
        session.setActive(true);
        narrator.addListener(this);
    }

    @Override public int onStartCommand(Intent i, int flags, int startId) {
        if (i != null && i.getStringExtra("cmd") != null) {
            switch (i.getStringExtra("cmd")) {
                case "toggle": narrator.toggle(); break;
                case "next": narrator.next(); break;
                case "prev": narrator.prev(); break;
                case "close": narrator.close(); stopSelf(); return START_NOT_STICKY;
                default: break;
            }
        }
        changed();
        return START_NOT_STICKY;
    }

    @Override public void onDestroy() {
        narrator.removeListener(this);
        session.release();
        super.onDestroy();
    }

    @Override public void changed() {
        Doc d = narrator.doc();
        if (d == null) {
            stopForeground(STOP_FOREGROUND_REMOVE);
            stopSelf();
            return;
        }
        boolean playing = narrator.playing();
        int n = Math.max(1, d.sentences.size()), at = Math.min(narrator.position(), n - 1);
        String now = d.sentences.isEmpty() ? "" : d.sentences.get(at);
        session.setMetadata(new MediaMetadata.Builder()
                .putString(MediaMetadata.METADATA_KEY_TITLE, d.title)
                .putString(MediaMetadata.METADATA_KEY_ARTIST, "SteerTTS · " + Voices.get(this).nameOf(Style.load(this).voice, null))
                .putString(MediaMetadata.METADATA_KEY_DISPLAY_SUBTITLE, now)
                .putLong(MediaMetadata.METADATA_KEY_DURATION, n * 1000L)   // one "second" per sentence: the seek bar is the book
                .build());
        session.setPlaybackState(new PlaybackState.Builder()
                .setActions(PlaybackState.ACTION_PLAY | PlaybackState.ACTION_PAUSE | PlaybackState.ACTION_PLAY_PAUSE
                        | PlaybackState.ACTION_SKIP_TO_NEXT | PlaybackState.ACTION_SKIP_TO_PREVIOUS
                        | PlaybackState.ACTION_STOP | PlaybackState.ACTION_SEEK_TO)
                .setState(playing ? PlaybackState.STATE_PLAYING : PlaybackState.STATE_PAUSED, at * 1000L, 0f)
                .build());
        Notification note = notification(d, now, playing);
        if (playing) {
            if (Build.VERSION.SDK_INT >= 29) startForeground(ID, note, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK);
            else startForeground(ID, note);
        } else {
            stopForeground(STOP_FOREGROUND_DETACH);          // paused: the notification stays, can be swiped away
            getSystemService(NotificationManager.class).notify(ID, note);
        }
    }

    private PendingIntent cmd(String c, int code) {
        return PendingIntent.getService(this, code, new Intent(this, ReaderService.class).putExtra("cmd", c),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
    }

    private Notification notification(Doc d, String now, boolean playing) {
        NotificationManager nm = getSystemService(NotificationManager.class);
        NotificationChannel ch = new NotificationChannel(CHANNEL, "Reading aloud", NotificationManager.IMPORTANCE_LOW);
        ch.setShowBadge(false);
        nm.createNotificationChannel(ch);
        return new Notification.Builder(this, CHANNEL)
                .setSmallIcon(R.drawable.ic_launcher_monochrome)
                .setContentTitle(d.title)
                .setContentText(now)
                .setContentIntent(session.getController().getSessionActivity())
                .setDeleteIntent(cmd("close", 4))
                .setOngoing(playing)
                .setVisibility(Notification.VISIBILITY_PUBLIC)
                .addAction(new Notification.Action.Builder(null, "Previous", cmd("prev", 1)).build())
                .addAction(new Notification.Action.Builder(null, playing ? "Pause" : "Play", cmd("toggle", 2)).build())
                .addAction(new Notification.Action.Builder(null, "Next", cmd("next", 3)).build())
                .setStyle(new Notification.MediaStyle().setMediaSession(session.getSessionToken()).setShowActionsInCompactView(0, 1, 2))
                .build();
    }
}

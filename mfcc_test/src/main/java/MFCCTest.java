import be.tarsos.dsp.AudioDispatcher;
import be.tarsos.dsp.AudioEvent;
import be.tarsos.dsp.AudioProcessor;
import be.tarsos.dsp.io.jvm.AudioDispatcherFactory;
import be.tarsos.dsp.io.UniversalAudioInputStream;
import be.tarsos.dsp.io.TarsosDSPAudioFormat;
import be.tarsos.dsp.mfcc.MFCC;
import be.tarsos.dsp.io.PipedAudioStream;
import be.tarsos.dsp.io.TarsosDSPAudioInputStream;
import java.util.*;
import java.io.*;
public class MFCCTest {

//	private static int counter = 0;

	//@Test
	public List<float[]> Test() {
		String path = "data/test.wav";
		int sampleRate = 16000;
		int bufferSize = 1024;
		int bufferOverlap = 128;
		//final float[] floatBuffer = TestUtilities.audioBufferSine();
		// PipedAudioStream file = new PipedAudioStream("test.wav");
		// TarsosDSPAudioInputStream stream = file.getMonoStream(16000,0);
		// System.out.println(stream);
		InputStream inStream = null;
		File f = new File(path);
		try{
         inStream = new FileInputStream(f);
      	}
      	catch (FileNotFoundException ex) {
        }

		//AudioDispatcher dispatcher = new AudioDispatcher(stream, bufferSize, bufferOverlap);
		final List<float[]>mfccList = new ArrayList<>(200);
        AudioDispatcher dispatcher = new AudioDispatcher(new UniversalAudioInputStream(inStream, new TarsosDSPAudioFormat(sampleRate, 16, 1, true, true)), bufferSize, bufferOverlap);
		// //final AudioDispatcher dispatcher = AudioDispatcherFactory.fromPipe("test.wav", sampleRate, bufferSize, bufferOverlap);
		final MFCC mfcc = new MFCC(bufferSize, sampleRate, 20, 50, 25, 200000);
		dispatcher.addAudioProcessor(mfcc);

		dispatcher.addAudioProcessor(new AudioProcessor() {
			@Override
			public boolean process(AudioEvent audioEvent) {
				System.out.println(Arrays.toString(mfcc.getMFCC()));
				mfccList.add(mfcc.getMFCC());
				return true;
			}
			@Override
			public void processingFinished() {
				System.out.println("enter processingFinished");
			}
		});
		dispatcher.run();
		System.out.println(dispatcher.getFormat());
		System.out.println("mfcc size: " + mfccList.size());
		return mfccList;
	}
	public void floatArrayPrint(float [] farray){

		for(int i = 0; i < farray.length; i++){
			System.out.print(farray[i]);
		}
	}
	public static void main(String[] args){
		MFCCTest a = new MFCCTest();
		List<float[]> mfcc = a.Test();

	}

}

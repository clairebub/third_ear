package com.rj10.a3;

import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import java.util.List;

/**
 * Created on 12/10/17.
 */

public class RecognizedTextsAdapter extends
        RecyclerView.Adapter<RecognizedTextsAdapter.MyViewHolder> {

    public class MyViewHolder extends RecyclerView.ViewHolder {
        public TextView text;

        public MyViewHolder(View view) {
            super(view);
            text = (TextView) view.findViewById(R.id.text);
        }
    }

    private List<RecognizedText> textList;

    public RecognizedTextsAdapter(List<RecognizedText> textList) {
        this.textList = textList;
    }

    @Override
    public MyViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View itemView = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.speech_recog_row, parent, false);
        return new MyViewHolder(itemView);
    }

    @Override
    public void onBindViewHolder(MyViewHolder myViewHolder, int position) {
        RecognizedText recogText = textList.get(position);
        myViewHolder.text.setText(recogText.getText());
    }

    @Override
    public int getItemCount() {
        return textList.size();
    }


}

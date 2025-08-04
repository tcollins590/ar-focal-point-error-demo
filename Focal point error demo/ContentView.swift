//
//  ContentView.swift
//  Focal point error demo
//
//  Created by Tyler Collins on 8/4/25.
//

import SwiftUI

struct ContentView: View {
    @State private var showImmersiveView = false
    
    var body: some View {
        VStack {
            Button("Enter AR") {
                showImmersiveView = true
            }
            .font(.title)
            .padding()
            .background(Color.blue)
            .foregroundColor(.white)
            .cornerRadius(10)
        }
        .padding()
        .fullScreenCover(isPresented: $showImmersiveView) {
            ImmersiveView()
        }
    }
}

#Preview {
    ContentView()
}
